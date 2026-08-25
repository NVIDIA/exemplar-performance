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

"""install.py [<cluster>] [--dry-run]

New-cluster onboarding + SLURM-aligned recipe multi-select / bulk setup.

Two-tier flow (each phase printed as a panel):
  Phase 0  Cluster selection — ONLY when no cluster was given. Pick a profile from a numbered menu; skipped
           entirely when there is exactly one profile or $LLMB_CLUSTER names one. Not a tty → no prompt,
           just a clear "pass <cluster-profile>" error (CI/scripts must never hang here).
           `install <cluster> [--recipes ...]` is completely unchanged — this is an extra door, not a swap.
  Phase A' Already-installed panel — what THIS cluster already has, read from the cluster (not local stamps)
           via fleet_render.discover_model_caches / discover_installed, with fleet's honest verdicts: a
           Bound-but-unvouched-for PVC reads `contents UNVERIFIED`, never ✓, and an unreadable cluster says
           nothing at all rather than "absent".
  Phase A  Profile review + confirm — resolved profile state + cluster health; ask to proceed.
  Phase B  Recipe/cell multi-select — list catalog cells whose gpu_type matches the cluster GPU_PRODUCT
           (INCLUDING wip, status-marked); already-installed cells (per the per-cluster install stamp) render
           GREYED-OUT / non-selectable. Numbered comma-select widget (Teleport/non-tty safe, no questionary).
  Phase C  Model cache — download the UNION of the models the selected cells need (dedup), reusing the
           existing model-selection + download-Job path. --all-matching auto-downloads the union.
  Phase D  Per-cell bulk setup — for each selected cell: stage the benchmark dataset, run
           preflight.py <cell> <cluster>, and emit the install stamp. Stop at the first blocker for each cell.

Install stamp (per-cluster, gitignored, 0600): cluster-profiles/.state/<cluster>.install.jsonl — one JSONL
line per cell set up here (cell, recipe_hash, model_repo, staged, preflight, job_mode, stamped_at). This is
what makes install idempotent (greyed-out) and feeds `fleet --stages`' per-cluster INSTALLED inventory.
Written by BOTH origins: Phase D above, AND the `run` inline-stage path (run.sh shells out to
`install.py --record-stage …` → record_stage_stamp() → write_install_stamp), so a cell staged by `run` — not
only by `install` — shows up in fleet's INSTALLED section, and a stage FAILURE is recorded there too.

Usage:
  scripts/install.py                                   interactive: pick cluster → pick recipes
  scripts/install.py <cluster>                         interactive two-tier install
  scripts/install.py --cluster <cluster>               same (named flag alias)
  scripts/install.py <cluster> --list-recipes          print GPU-compat matrix + installed set, exit (offline)
  scripts/install.py <cluster> --recipes cellA,cellB   headless: set up these cells (already-installed
                                                       ones are a NO-OP; --reinstall redoes them)
  scripts/install.py <cluster> --all-matching          headless: set up every matching cell, auto-download union
  scripts/install.py <cluster> --skip-model-download   stage + preflight only
  scripts/install.py <cluster> --dry-run               show plan without applying anything

The download Job is rendered from serving/download/templates/model-download.yaml.j2.
Each Job is idempotent (the Job checks a PVC sentinel before downloading).
Models with an unpinned model_revision are refused (no sha256 → no comparability guarantee).

Reuses: profile_resolver.list_profiles(), resolve(), profile_env_path(), _read_env(), check_target_compat(),
        _norm_gpu(); lane.resolve_lane(); preflight.py; the stage-*.sh scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

try:
    import yaml
except ImportError:
    sys.exit("requires: pip install pyyaml")

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    sys.exit("requires: pip install jinja2")

import model_cache as _mc  # noqa: E402
import profile_resolver as _pr  # noqa: E402

DOWNLOAD_TMPL = ROOT / "serving" / "download" / "templates" / "model-download.yaml.j2"
CATALOG_PATH = ROOT / "catalog.json"

# Environment variables that authorize expensive operations accept only explicit boolean values.
# Unrecognized values never grant permission.
_ENV_TRUE = {"1", "true", "yes", "on"}
_ENV_FALSE = {"", "0", "false", "no", "off"}


def _env_flag(name: str, default: bool = False) -> bool:
    """PURE-ish — read an authorising env var. Only an explicit true-word enables; anything unrecognised
    warns and falls back to `default`, because a typo must not silently grant permission.
    """
    raw = (os.environ.get(name) or "").strip().lower()
    if raw in _ENV_TRUE:
        return True
    if raw in _ENV_FALSE:
        return False
    print(
        f"  ⚠ {name}={raw!r} is not a recognised boolean — treating it as "
        f"{'ENABLED' if default else 'DISABLED'}. Use 1/true/yes/on or 0/false/no/off.",
        file=sys.stderr,
    )
    return default


# Per-cluster install stamp (gitignored). One JSONL line per (cell) set up on this cluster; makes the recipe
# multi-select idempotent (already-installed cells render greyed-out) and feeds the future `fleet --grid`.
STATE_DIR = _pr.PROFILES_DIR / ".state"

# UX vocab (ASCII-clean, no 🟡). Shared with the SLURM express flow's status words.
GLYPH_READY = "✓"  # ready / done / pass
GLYPH_NEEDS = "⚠"  # needs operator input
GLYPH_BLOCKED = "❌"  # blocked by a fixable failure
GLYPH_INSTALLED = "🔒"  # greyed-out — already installed on this cluster at the current recipe_hash

# Known approximate model sizes (GiB) for display in Phase B.
# Keyed on the HuggingFace model_repo (normalized to lowercase).
_KNOWN_SIZES_GIB: dict[str, int] = {
    "nvidia/nvidia-nemotron-3-ultra-550b-a55b-nvfp4": 400,
    "zai-org/glm-5-fp8": 704,
    "nvidia/llama-3.1-nemotron-nano-8b-v1": 15,
}

# --- Model-cache auto-sizing -------------------------------------------------
# Cache sizing uses the declared or known model size plus headroom for the Hugging Face layout,
# datasets, and partial downloads.
_CACHE_FLOOR_GIB = 100  # last-resort floor when NO size can be derived (unknown model, nothing declared)
_CACHE_HEADROOM_FRAC = 0.25  # +25% over the raw model bytes ...
_CACHE_HEADROOM_MIN_GIB = 50  # ... but at least +50Gi
_CACHE_UNDERSIZE_FRAC = 0.05  # tolerance for a PRE-EXISTING PVC that is marginally (≤5%) smaller than needed
# Large models default to the configured shared RWX cache class; smaller models default to the configured
# RWO block class. Explicit recipe or profile settings take precedence. Unknown models should declare size
# and access mode when the defaults are unsuitable.
_FAST_CACHE_MODEL_GIB = 50

# Discovery may suggest a pre-provisioned shared RWX cache, but the cluster profile remains authoritative.
# Names and labels are configurable through LLMB_SHARED_CACHE_NAMES and LLMB_SHARED_CACHE_LABEL.
_SHARED_CACHE_NAMES = ("shared-model-cache",)
_SHARED_CACHE_LABEL = "llmb.nvidia.com/model-cache"  # value ignored; presence marks a shared cache
# Ranking floor for the large-RWX-cache suggestion. Candidate listing remains size-independent.
_SHARED_CACHE_MIN_RWX_GIB = 1024  # a Bound RWX PVC ≥ 1Ti whose name looks like a cache
_SHARED_CACHE_NAME_HINTS = (
    "model-cache",
    "model-weights",
    "weights",
    "hf-cache",
    "huggingface",
)
# Candidate listing uses broad name hints so operators can select a suitable existing cache.
_CACHE_CANDIDATE_HINTS = ("cache", "weights", "huggingface", "models")

# Cap the wait for a newly created cache PVC. WaitForFirstConsumer PVCs may remain Pending until scheduled;
# only explicit provisioning or attachment errors are treated as failures.
_CACHE_BIND_BUDGET_S = 90

# Maximum download size allowed when free capacity cannot be measured. Larger writes require a successful
# capacity check or an explicit operator override.
_UNMEASURED_DOWNLOAD_MAX_GIB = 50


# ---------------------------------------------------------------------------
# Injectable kubectl runner (pure functions accept this as a parameter so
# selftest_onboarding.py can inject a mock without touching a cluster).
# ---------------------------------------------------------------------------


def make_krun(context: str = ""):
    """Build a kubectl runner pinned to `context` (empty = current-context). Threads `--context` into every
    call so `llmb-k8s install` targets the PROFILE's cluster, never the global current-context — otherwise a
    profile pinned to prod could probe/download/apply against whatever context kubectl happens to have.
    """
    ctx_args = ["--context", context] if context else []

    def _krun(args: list[str], timeout: int = 30) -> tuple[int, str, str]:
        try:
            p = subprocess.run(
                ["kubectl", *ctx_args, "--request-timeout=25s", *args],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return p.returncode, p.stdout, p.stderr
        except Exception as e:
            return 1, "", str(e)

    return _krun


# Default runner (no context pinned) — the injectable default for pure functions + tests.
default_krun = make_krun("")


def make_manifest_applier(context: str = ""):
    """Build a `kubectl apply -f -` runner (stdin manifest) pinned to `context`. Separate from krun because
    krun has no stdin channel; mirrors the download-Job apply (same context-pinning discipline). Returns a
    callable `(manifest, ns) -> (rc, stderr)`. Injectable so selftests can capture the manifest without a
    cluster."""
    ctx_args = ["--context", context] if context else []

    def _apply(manifest: str, ns: str) -> tuple[int, str]:
        try:
            p = subprocess.run(
                ["kubectl", *ctx_args, "-n", ns, "apply", "-f", "-"],
                input=manifest,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return p.returncode, p.stderr
        except Exception as e:
            return 1, str(e)

    return _apply


# Default applier (no context pinned) — injectable default for pure functions + tests.
default_apply = make_manifest_applier("")


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def load_catalog(catalog_path: Path = CATALOG_PATH) -> list[dict]:
    """Load catalog.json; return [] if missing or malformed."""
    if not catalog_path.exists():
        return []
    try:
        return json.loads(catalog_path.read_text())
    except Exception:
        return []


def _norm_repo(repo: str) -> str:
    return (repo or "").strip().lower()


def catalog_models(catalog: list[dict], root: Path = ROOT) -> list[dict]:
    """Return a deduplicated list of models referenced by catalog recipes.

    Each entry:
        model_name    short model name (from catalog 'model' field)
        model_repo    HuggingFace repo id  (from recipe.yaml serving.model_repo)
        model_revision  pinned revision or ""
        recipe_count  number of catalog cells that use this model
        cell_paths    list of _path values

    Models with no serving.model_repo (recipe.yaml missing or unparseable) are skipped.
    """
    by_repo: dict[str, dict] = {}
    for item in catalog:
        path = item.get("_path", "")
        recipe_file = root / path / "recipe.yaml"
        if not recipe_file.exists():
            continue
        try:
            r = yaml.safe_load(recipe_file.read_text()) or {}
        except Exception:
            continue
        s = r.get("serving") or {}
        model_repo = (s.get("model_repo") or "").strip()
        if not model_repo:
            continue
        model_revision = (s.get("model_revision") or "").strip()
        model_name = item.get("model", "") or model_repo.split("/")[-1]
        key = _norm_repo(model_repo)
        if key not in by_repo:
            by_repo[key] = {
                "model_name": model_name,
                "model_repo": model_repo,
                "model_revision": model_revision,
                "recipe_count": 0,
                "cell_paths": [],
            }
        # If any cell has a pinned revision, prefer it. Warn on a CONFLICT (two cells share a repo but pin
        # different revisions) — first-seen wins silently would otherwise stage the wrong weights (audit #6).
        existing_rev = by_repo[key]["model_revision"]
        if model_revision and not existing_rev:
            by_repo[key]["model_revision"] = model_revision
        elif model_revision and existing_rev and model_revision != existing_rev:
            print(
                f"  ⚠ revision conflict for {model_repo}: cell pins {model_revision} but "
                f"{existing_rev} already chosen (from {by_repo[key]['cell_paths'][0]}) — "
                f"installing {existing_rev}; the other cell will not match on the PVC."
            )
        by_repo[key]["recipe_count"] += 1
        by_repo[key]["cell_paths"].append(path)
    return list(by_repo.values())


def model_size_gib(model: dict) -> int | None:
    """Return approximate download size in GiB, or None if unknown."""
    return _KNOWN_SIZES_GIB.get(_norm_repo(model["model_repo"]))


_QTY_BIN = {"Ki": 1, "Mi": 2, "Gi": 3, "Ti": 4, "Pi": 5, "Ei": 6}
_QTY_DEC = {
    "K": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
    "E": 1000**6,
}


def _parse_quantity_gib(s: str) -> int | None:
    """PURE — parse a k8s storage quantity ('500Gi', '1.2Ti', '800G', '100Gi', bare bytes) → integer GiB
    (ceil). None if empty/unparseable. Handles binary (Ki/Mi/Gi/Ti…) and decimal (K/M/G/T…) suffixes.
    """
    s = (s or "").strip()
    if not s:
        return None
    m = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([EPTGMK]i?|)", s)
    if not m:
        return None
    val, unit = float(m.group(1)), m.group(2)
    if unit in _QTY_BIN:
        bytes_ = val * (1024 ** _QTY_BIN[unit])
    elif unit in _QTY_DEC:
        bytes_ = val * _QTY_DEC[unit]
    else:  # bare number = bytes
        bytes_ = val
    return max(1, math.ceil(bytes_ / (1024**3)))


def _fmt_gib(n: int) -> str:
    """PURE — integer GiB → a k8s quantity string."""
    return f"{int(n)}Gi"


def _auto_cache_size_gib(model_sizes_gib: list[int | None]) -> int | None:
    """PURE — GiB a cache needs to FIT the given model(s) + headroom. Models that SHARE one cache are
    SUMMED (dedup-share). None if NO size is known (caller must then fall back + warn, never silently
    undersize). Headroom = max(+25%, +50Gi)."""
    known = [s for s in model_sizes_gib if s and s > 0]
    if not known:
        return None
    total = sum(known)
    headroom = max(math.ceil(total * _CACHE_HEADROOM_FRAC), _CACHE_HEADROOM_MIN_GIB)
    return total + headroom


def _cell_model_size_gib(cell: dict, root: Path = ROOT) -> int | None:
    """On-disk model GiB for a catalog cell, via its recipe.yaml serving.model_repo → model_size_gib
    (the SAME table the download space-check uses). None if the repo is missing or unlisted. Reads disk
    (like cache_pvc_by_repo); never touches a cluster."""
    rp = root / (cell.get("_path") or "") / "recipe.yaml"
    try:
        r = yaml.safe_load(rp.read_text()) or {}
    except Exception:
        return None
    repo = ((r.get("serving") or {}).get("model_repo") or "").strip()
    if not repo:
        return None
    return model_size_gib({"model_repo": repo})


def _cell_model_repo(cell: dict, root: Path = ROOT) -> str | None:
    """The normalized serving.model_repo for a cell — the identity of the WEIGHTS ON DISK, so two cells
    naming the same repo share one copy and cache sizing must count it once.

    THREE-STATE: the repo when pinned, '' when the recipe is readable but pins none, and None when the
    recipe could NOT be read. Collapsing the last two would make an unreadable recipe.yaml claim "this cell
    has no model", and the size dedupe would then silently merge it with every other unreadable cell —
    under-sizing a claim is exactly the failure this grouping exists to prevent."""
    rp = root / (cell.get("_path") or "") / "recipe.yaml"
    try:
        r = yaml.safe_load(rp.read_text()) or {}
    except Exception:
        return None
    return _norm_repo(((r.get("serving") or {}).get("model_repo") or "").strip())


def _cell_model_revision(cell: dict, root: Path = ROOT) -> str | None:
    """The pinned serving.model_revision for a catalog cell.

    Returns a revision string, an empty string for a readable unpinned recipe, or `None` when the recipe
    cannot be read. Callers preserve unknown separately from absence."""
    rp = root / (cell.get("_path") or "") / "recipe.yaml"
    try:
        r = yaml.safe_load(rp.read_text()) or {}
    except Exception:
        return None
    return str(((r.get("serving") or {}).get("model_revision") or "")).strip()


# Download Job deadlines scale with model size while retaining a minimum timeout.
_DOWNLOAD_DEADLINE_FLOOR_S = 7200  # 2h floor — keeps small models fast-failing on a genuine hang
_DOWNLOAD_SECONDS_PER_GIB = 45  # ~1.3 GiB/min sustained incl. headroom (slow-link safe)


def download_deadline_s(size_gib: int | None) -> int:
    """PURE. activeDeadlineSeconds for a model's download Job, sized from its on-disk GiB (with headroom).
    Unknown size → the 2h floor. Testable without a cluster."""
    if not size_gib or size_gib <= 0:
        return _DOWNLOAD_DEADLINE_FLOOR_S
    return max(_DOWNLOAD_DEADLINE_FLOOR_S, int(size_gib) * _DOWNLOAD_SECONDS_PER_GIB)


# Download Job memory. huggingface_hub snapshot_download + hf_transfer buffer shards in RAM; that working
# set scales with download PARALLELISM × chunk size, NOT total model size — so a flat, generous FLOOR is the
# right shape. The old req 4Gi / limit 8Gi OOMKilled a ~400 GB model (nemotron-ultra-3) download, and with
# backoffLimit=0 that's terminal (QA #). 32Gi limit / 8Gi request is a safe floor for any model size.
_DOWNLOAD_MEM_REQUEST_GIB = 8
_DOWNLOAD_MEM_LIMIT_GIB = 32


def download_mem_limit_gib(size_gib: int | None = None) -> int:
    """PURE — memory LIMIT (GiB) for a model's download Job. A flat 32Gi floor (hf_transfer RAM scales with
    parallelism×chunk, not model size); `size_gib` is accepted for future scaling but never lowers the floor.
    Testable without a cluster."""
    return _DOWNLOAD_MEM_LIMIT_GIB


def download_mem_request_gib(size_gib: int | None = None) -> int:
    """PURE — memory REQUEST (GiB) for a model's download Job. Flat 8Gi floor."""
    return _DOWNLOAD_MEM_REQUEST_GIB


# ---------------------------------------------------------------------------
# Provision progress / ETA notes (QA fix #4) — a one-line "▸ <stage> — ~N min (<what's happening>)" banner
# so a user can tell healthy-slow from hung. Coarse ranges are fine; the point is a SIGNAL, not precision.
# Both helpers are PURE (testable without a cluster).
# ---------------------------------------------------------------------------


def _stage_note(stage: str, eta: str, detail: str) -> str:
    """PURE. Format a progress/ETA banner line: '▸ <stage> — <eta>  (<what's happening>)'."""
    return f"  ▸ {stage} — {eta}  ({detail})"


def download_eta_text(size_gib: int | None) -> str:
    """PURE. Coarse, friendly ETA range for a model download of `size_gib` GiB (None → unknown). Bucketed
    so the user gets a signal ('is this normal?') without over-promising precision."""
    if not size_gib or size_gib <= 0:
        return "~a few min (size unknown)"
    if size_gib <= 10:
        return "~2-5 min"
    if size_gib <= 50:
        return "~5-15 min"
    if size_gib <= 200:
        return "~15-45 min"
    return "~45+ min (very large — multi-hundred-GiB weights)"


# ---------------------------------------------------------------------------
# Recipe/cell selection (Phase B) — pure catalog + profile filtering
# ---------------------------------------------------------------------------


def cell_id(cell: dict) -> str:
    """Stable, human-facing identifier for a cell — the catalog 'name', falling back to the _path basename.
    `--recipes` also accepts the catalog-relative or absolute cell path; the stamp remains keyed by _path.
    Pure."""
    return (cell.get("name") or "").strip() or Path(cell.get("_path", "")).name


def gpu_matching_cells(catalog: list[dict], prof: dict) -> list[dict]:
    """Return catalog cells whose gpu_type matches the cluster's GPU_PRODUCT (INCLUDING wip — the owner wants
    devs to see everything; releases won't ship wip). Match is _norm_gpu-normalized so 'NVIDIA-GB200' and
    'GB200' compare equal (profile_resolver._norm_gpu / profile_gpu_type). Pure — no cluster calls.

    Sorted by (scenario, distribution, cell id) so the numbered list is stable across runs.
    """
    have = _pr.profile_gpu_type(prof)
    matched = [c for c in catalog if have and _pr._norm_gpu(c.get("gpu_type", "")) == have]
    return sorted(
        matched,
        key=lambda c: (c.get("scenario", ""), c.get("distribution", ""), cell_id(c)),
    )


def _stamp_ready(st: dict) -> bool:
    """A stamp represents a READY cell (→ greyed-out) only if setup actually completed: staging all-ok and
    preflight did not FAIL. A recorded fail/needs-input stamp is kept for the audit trail + fleet grid but
    must NOT grey the cell out (re-setup is still due). Pure."""
    if (st.get("preflight") or "").lower() == "fail":
        return False
    staged = st.get("staged") or {}
    # staged is {step: {"ok": bool, "sha": ...}}; any explicit ok=False → not ready.
    for v in staged.values():
        if isinstance(v, dict) and v.get("ok") is False:
            return False
    return True


def cell_installed(cell: dict, stamps: dict[str, dict]) -> bool:
    """True if this cluster's install stamp records this cell READY at the CURRENT recipe_hash (→ greyed-out,
    non-selectable). A stamp at a stale recipe_hash, or one whose staging failed / preflight FAILed, does NOT
    count — the recipe moved or setup didn't complete, so re-setup is due."""
    st = stamps.get(cell.get("_path", ""))
    if not st:
        return False
    if not (cell.get("recipe_hash") and st.get("recipe_hash") == cell.get("recipe_hash")):
        return False
    return _stamp_ready(st)


# ---------------------------------------------------------------------------
# Per-cluster install stamp (JSONL, gitignored, 0600) — idempotency + fleet grid
# ---------------------------------------------------------------------------


def install_stamp_path(cluster: str, state_dir: Path = STATE_DIR) -> Path:
    """Path to a cluster's install stamp file. One JSONL line per cell set up on this cluster."""
    return state_dir / f"{cluster}.install.jsonl"


def read_install_stamps(cluster: str, state_dir: Path = STATE_DIR) -> dict[str, dict]:
    """Read a cluster's install stamp → {cell_path: latest_stamp}. Last line per cell wins (append-only log).
    Missing/malformed file → {} (nothing installed yet). Never raises."""
    path = install_stamp_path(cluster, state_dir)
    if not path.exists():
        return {}
    stamps: dict[str, dict] = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            cell = rec.get("cell")
            if cell:
                stamps[cell] = rec  # later line wins
    except Exception:
        return stamps
    return stamps


def write_install_stamp(
    cluster: str,
    cell_path: str,
    recipe_hash: str,
    model_repo: str,
    staged: dict,
    preflight: str,
    job_mode: str,
    state_dir: Path = STATE_DIR,
) -> dict:
    """Append one install-stamp line for a cell (atomic append, 0600 dir+file). Returns the written record.

    Fields (per the design): cell, recipe_hash, model_repo, staged (dataset status and sha),
    preflight (pass/warn/fail), job_mode, stamped_at."""
    state_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(state_dir, 0o700)
    except OSError:
        pass
    rec = {
        "cell": cell_path,
        "recipe_hash": recipe_hash,
        "model_repo": model_repo,
        "staged": staged,
        "preflight": preflight,
        "job_mode": job_mode,
        "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = install_stamp_path(cluster, state_dir)
    # Atomic append: open O_APPEND with 0600 perms, write one line, fsync. Concurrent installs on the same
    # cluster (different shells) each append their own line — no lost update, last-writer-wins on read.
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(rec, separators=(",", ":")) + "\n").encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return rec


def _cell_relpath(cell: str, root: Path = ROOT) -> str:
    """Normalize a cell reference (an absolute dir, a ./-relative dir, or an already-relative _path) to the
    repo-relative _path the catalog + fleet key on (e.g. 'recipes/llm-perf/1m/...'). Pure. Falls back to the
    raw string (trailing slash stripped) when the path is not under `root`."""
    p = Path(cell)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return str(cell).rstrip("/")


def _cell_meta(cell_path: str, catalog: list[dict] | None = None, root: Path = ROOT) -> tuple[str, str]:
    """Best-effort (recipe_hash, model_repo) for a repo-relative cell _path — from the catalog (recipe_hash)
    and recipe.yaml (serving.model_repo). Missing/unreadable → ('', ''). Pure. Keeps the stamp's schema in
    lockstep with what install writes so fleet's install-stamp reader treats both origins identically.
    """
    recipe_hash = ""
    if catalog is None:
        catalog = load_catalog()
    for item in catalog or []:
        if item.get("_path") == cell_path:
            recipe_hash = item.get("recipe_hash", "") or ""
            break
    model_repo = ""
    recipe_file = root / cell_path / "recipe.yaml"
    if recipe_file.exists():
        try:
            r = yaml.safe_load(recipe_file.read_text()) or {}
            model_repo = ((r.get("serving") or {}).get("model_repo") or "").strip()
        except Exception:
            model_repo = ""
    return recipe_hash, model_repo


def record_stage_stamp(
    cell: str,
    cluster: str,
    *,
    step_key: str,
    staged: str,
    preflight: str,
    sha: str = "",
    job_mode: str = "",
    state_dir: Path = STATE_DIR,
    catalog: list[dict] | None = None,
    root: Path = ROOT,
) -> dict:
    """Record ONE stage attempt into the per-cluster install stamp, from ANY staging origin (the `run` inline
    stage path, manual staging, or install). Reuses write_install_stamp so the JSONL schema is identical to
    what `llmb-k8s install` writes and what fleet reads. `staged` ∈ {ok, failed, skipped}; `preflight` ∈
    {pass, warn, fail, skipped}. FAILURES are recorded too (staged=failed / preflight=fail) so a broken install
    is VISIBLE in fleet, not silent. Cheap + idempotent (append-only; last line per cell wins). Pure given the
    injected paths/catalog (offline-testable)."""
    cell_path = _cell_relpath(cell, root=root)
    recipe_hash, model_repo = _cell_meta(cell_path, catalog=catalog, root=root)
    staged_ok = (
        {"ok": True} if staged == "ok" else {"ok": False} if staged == "failed" else {"ok": None, "reason": "skipped"}
    )
    if sha and staged_ok.get("ok"):
        staged_ok["sha"] = sha
    return write_install_stamp(
        cluster=cluster,
        cell_path=cell_path,
        recipe_hash=recipe_hash,
        model_repo=model_repo,
        staged={step_key or "stage": staged_ok},
        preflight=preflight,
        job_mode=job_mode,
        state_dir=state_dir,
    )


def _cli_record_stage(argv: list[str]) -> int:
    """`install.py --record-stage <cell> <cluster> --step <k> --staged <ok|failed|skipped> --preflight <...>`.
    A thin, dependency-light entry the `run` path (run.sh) shells out to after it stages a cell inline, so a
    cell staged by `run` (not just by `install`) shows up in fleet's INSTALLED inventory. Best-effort by
    design — never fails a run: prints one terse line and returns 0 even on a write error.
    """
    ap = argparse.ArgumentParser(prog="install.py --record-stage", add_help=True)
    ap.add_argument("cell")
    ap.add_argument("cluster")
    ap.add_argument("--step", default="stage", help="stage step key (e.g. stage-dataset)")
    ap.add_argument("--staged", choices=("ok", "failed", "skipped"), default="ok")
    ap.add_argument("--preflight", choices=("pass", "warn", "fail", "skipped"), default="pass")
    ap.add_argument("--sha", default="")
    ap.add_argument("--job-mode", default="")
    a = ap.parse_args(argv)
    try:
        rec = record_stage_stamp(
            a.cell,
            a.cluster,
            step_key=a.step,
            staged=a.staged,
            preflight=a.preflight,
            sha=a.sha,
            job_mode=a.job_mode,
        )
        print(
            f"  install-stamp: {rec['cell']} → staged={a.staged} preflight={a.preflight} "
            f"({install_stamp_path(a.cluster)})"
        )
    except Exception as e:  # a stamp write must NEVER break a run
        print(f"  install-stamp: skipped ({e})")
    return 0


def _cli_resolve_cache(argv: list[str]) -> int:
    """`install.py --resolve-cache <cell-dir> <profile.env>` — thin alias for
    `model_cache.py resolve`, kept so the flag documented on install stays valid."""
    import model_cache as _mc

    return _mc._cli_resolve(argv)


# ---------------------------------------------------------------------------
# Model union (Phase C) — dedup the models the selected cells need
# ---------------------------------------------------------------------------


def models_for_cells(cells: list[dict], root: Path = ROOT) -> list[dict]:
    """UNION of the models the selected cells reference, deduplicated by HF repo. Reuses catalog_models() so
    the revision-conflict warning + dedup behave identically to the whole-catalog path. Pure (reads recipe
    files under `root`)."""
    return catalog_models(cells, root=root)


# ---------------------------------------------------------------------------
# PVC probing (cluster calls; injectable via krun parameter)
# ---------------------------------------------------------------------------


def probe_pvc_free_gib(
    ns: str,
    pvc: str,
    krun=default_krun,
    prof: dict | None = None,
    why: list | None = None,
) -> float | None:
    """Spawn a transient pod, measure free space on the PVC, return GiB or None.

    `prof` supplies the cache-mounting PLACEMENT (nodeSelector + tolerations). Without it this pod lands
    anywhere, and on a cluster where only some nodes can mount the claim that is a coin flip — the reading
    then comes back None and the caller reports "capacity UNKNOWN" for a cache that is perfectly healthy.
    `why` (a list) receives the classified failure so the caller can say WHICH failure it was rather than
    collapsing unschedulable / mount-failed / RBAC-denied into one silent None."""
    prof = prof or {}
    sel, tol, _ = _mc.cache_pod_placement(prof)
    pod = f"llmb-install-probe-{int(time.time()) % 100000}"
    spec = {
        "restartPolicy": "Never",
        "tolerations": tol,
        "volumes": [{"name": "c", "persistentVolumeClaim": {"claimName": pvc}}],
        "containers": [
            {
                "name": "p",
                "image": "busybox:1.36",
                "command": ["sleep", "120"],
                "volumeMounts": [{"name": "c", "mountPath": "/cache"}],
            }
        ],
    }
    if sel:
        spec["nodeSelector"] = sel
    overrides = json.dumps({"spec": spec})
    try:
        _rc_c, _, _err_c = krun(
            [
                "-n",
                ns,
                "run",
                pod,
                "--image=busybox:1.36",
                "--restart=Never",
                f"--overrides={overrides}",
            ],
            timeout=30,
        )
        rc, _, _ = krun(
            ["-n", ns, "wait", "pod", pod, "--for=condition=ready", "--timeout=60s"],
            timeout=70,
        )
        if rc != 0:
            if why is not None:
                _, _ev, _ = krun(
                    [
                        "-n",
                        ns,
                        "get",
                        "events",
                        "--field-selector",
                        f"involvedObject.name={pod}",
                        "-o",
                        "jsonpath={.items[*].message}",
                    ],
                    timeout=20,
                )
                _, _ph, _ = krun(
                    ["-n", ns, "get", "pod", pod, "-o", "jsonpath={.status.phase}"],
                    timeout=20,
                )
                code, reason = _mc.classify_mounter_failure(_ph or "", _ev or "", _err_c or "")
                why.append((code, reason, _mc.cache_placement_hint(prof)))
            return None
        rc, out, _ = krun(["-n", ns, "exec", pod, "--", "df", "-k", "/cache"], timeout=30)
        if rc != 0 or not out:
            return None
        # df -k output: [Filesystem] 1K-blocks Used Available Use% Mounted-on
        # A long filesystem name makes df WRAP it onto its own line, so the /cache data line can
        # drop the leading Filesystem column — then a fixed parts[3] lands on Use% ("30%") and
        # int() blows up (QA #43). Index from the RIGHT, where the [Available, Use%, mount] tail
        # is stable whether or not the name wrapped; tolerate a stray '%' and non-numeric.
        lines = [ln for ln in out.splitlines() if "/cache" in ln]
        if not lines:
            return None
        parts = lines[0].split()
        if len(parts) >= 3:
            try:
                avail_kb = int(parts[-3].rstrip("%"))
            except ValueError:
                return None
            return avail_kb / (1024 * 1024)  # KiB → GiB
        return None
    finally:
        # Wait for deletion so short-lived probe pods do not accumulate in the namespace.
        krun(["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--now"], timeout=30)


def probe_pvc_capacity_gib(ns: str, name: str, krun=default_krun) -> int | None:
    """Requested/bound CAPACITY of a PVC in GiB (NOT free space) — used to detect a pre-existing cache
    claim that is smaller than a recipe now needs. Prefers the bound .status.capacity.storage, falling
    back to the requested .spec.resources.requests.storage. None on any failure. No transient pod needed.
    """
    for jp in ("{.status.capacity.storage}", "{.spec.resources.requests.storage}"):
        rc, out, _ = krun(["-n", ns, "get", "pvc", name, "-o", f"jsonpath={jp}"])
        if rc == 0 and out.strip():
            g = _parse_quantity_gib(out.strip())
            if g:
                return g
    return None


def probe_model_on_pvc(
    ns: str,
    pvc: str,
    model_repo: str,
    model_revision: str,
    model_cache_subpath: str = ".",
    krun=default_krun,
    prof: dict | None = None,
    why: list | None = None,
    facts_out: dict | None = None,
) -> str:
    """Is this model revision already on the PVC? Returns a model_cache STATE:
      complete | present-unverified | incomplete | absent | unknown

    `facts_out`, when given, is UPDATED with the raw probe facts. Callers that only want to know whether to
    download need the state; the caller that wants to write a PERMANENT sentinel needs the evidence behind
    it (see model_cache.sentinel_worthy) — a verdict alone cannot say whether the byte checks actually ran.

    The probe shares model_cache.cache_probe_script and cache_completeness with preflight, represents
    present-but-unverified weights explicitly, and reports scheduling, mount, and authorization failures.
    """
    if not model_revision:
        return _mc.STATE_UNKNOWN
    prof = prof or {}
    sel, tol, _ = _mc.cache_pod_placement(prof)
    pod = f"llmb-install-check-{int(time.time()) % 100000}"
    spec = {
        "restartPolicy": "Never",
        "tolerations": tol,
        "volumes": [{"name": "c", "persistentVolumeClaim": {"claimName": pvc, "readOnly": True}}],
        "containers": [
            {
                "name": "p",
                "image": "busybox:1.36",
                "command": ["sleep", "120"],
                "volumeMounts": [{"name": "c", "mountPath": "/cache", "readOnly": True}],
            }
        ],
    }
    if sel:
        spec["nodeSelector"] = sel
    overrides = json.dumps({"spec": spec})
    try:
        rc_c, _, err_c = krun(
            [
                "-n",
                ns,
                "run",
                pod,
                "--image=busybox:1.36",
                "--restart=Never",
                f"--overrides={overrides}",
            ],
            timeout=30,
        )
        rc, _, _ = krun(
            ["-n", ns, "wait", "pod", pod, "--for=condition=ready", "--timeout=60s"],
            timeout=70,
        )
        if rc != 0:
            if why is not None:
                _, ev, _ = krun(
                    [
                        "-n",
                        ns,
                        "get",
                        "events",
                        "--field-selector",
                        f"involvedObject.name={pod}",
                        "-o",
                        "jsonpath={.items[*].message}",
                    ],
                    timeout=20,
                )
                _, ph, _ = krun(
                    ["-n", ns, "get", "pod", pod, "-o", "jsonpath={.status.phase}"],
                    timeout=20,
                )
                code, reason = _mc.classify_mounter_failure(ph or "", ev or "", err_c or "")
                why.append((code, reason, _mc.cache_placement_hint(prof)))
            return _mc.STATE_UNKNOWN
        script = _mc.cache_probe_script(model_cache_subpath, model_repo, model_revision)
        rc, out, _ = krun(["-n", ns, "exec", pod, "--", "sh", "-c", script], timeout=45)
        if rc != 0:
            if why is not None:
                why.append(
                    (
                        "read-failed",
                        "the cache path could not be read inside the pod",
                        "",
                    )
                )
            return _mc.STATE_UNKNOWN
        facts = _mc.parse_cache_integrity_report(out)
        if facts_out is not None:
            facts_out.update(facts)
        state, reason = _mc.cache_completeness(facts)
        if why is not None:
            why.append((state, reason, ""))
        return state
    except Exception as e:
        if why is not None:
            why.append(("probe-error", str(e)[:160], ""))
        return _mc.STATE_UNKNOWN
    finally:
        # Wait for deletion so short-lived probe pods do not accumulate in the namespace.
        krun(["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--now"], timeout=30)


def stamp_download_sentinel(
    ns: str,
    pvc: str,
    model_repo: str,
    model_revision: str,
    model_cache_subpath: str = ".",
    krun=default_krun,
    prof: dict | None = None,
) -> tuple[bool, str]:
    """VERIFY-AND-STAMP: write the completion sentinel for a model this run has just PROVEN complete.

    Called only after model_cache.sentinel_worthy(facts) returned True — never on faith, and deliberately a
    STRICTER bar than the COMPLETE verdict itself. That asymmetry is the purpose: a verdict is re-derived
    from the disk on every probe, so a wrong one costs a single run; a sentinel is strong evidence for
    cache_completeness until contradictory physical evidence appears. The download Job now always resumes
    and verifies the immutable snapshot before rewriting that sentinel. A probe-created sentinel therefore
    still needs the stricter bar below, so install, preflight and fleet cannot certify a missing shard.

    What it converts is the honest case: a claim whose completeness had to be re-derived every time (the
    nemotron case: 113 shards, no .llmb_download_done directory at all, staged before the convention
    existed) into one that answers instantly and consistently for install, preflight and fleet.

    Needs a WRITE mount, so it uses the cache placement like every other mounting pod. Best-effort: a
    failure costs only the next re-verification, never correctness, so it never fails the install.
    """
    prof = prof or {}
    sel, tol, _ = _mc.cache_pod_placement(prof)
    sent = _mc.sentinel_path(model_cache_subpath, model_revision)
    pod = f"llmb-install-stamp-{int(time.time()) % 100000}"
    spec = {
        "restartPolicy": "Never",
        "tolerations": tol,
        "volumes": [{"name": "c", "persistentVolumeClaim": {"claimName": pvc}}],
        "containers": [
            {
                "name": "p",
                "image": "busybox:1.36",
                "command": [
                    "sh",
                    "-c",
                    f'mkdir -p "$(dirname /cache/{sent})" && '
                    f'printf "%s  %s\\n" verified-by-llmb-install {model_revision} > /cache/{sent}',
                ],
                "volumeMounts": [{"name": "c", "mountPath": "/cache"}],
            }
        ],
    }
    if sel:
        spec["nodeSelector"] = sel
    rc, _, err = krun(
        [
            "-n",
            ns,
            "run",
            pod,
            "--image=busybox:1.36",
            "--restart=Never",
            f"--overrides={json.dumps({'spec': spec})}",
        ],
        timeout=30,
    )
    if rc != 0:
        return (False, f"could not start the stamp pod: {(err or '').strip()[:120]}")
    try:
        # POLL for a TERMINAL phase. `kubectl wait --for=condition=Ready=false` returns immediately while
        # the pod is still Pending (a Pending pod is trivially not-Ready), which would report a healthy
        # stamp as "ended in phase Pending" — a false negative on a write we actually made.
        deadline = time.time() + 120
        ph = ""
        while time.time() < deadline:
            _, ph, _ = krun(
                ["-n", ns, "get", "pod", pod, "-o", "jsonpath={.status.phase}"],
                timeout=20,
            )
            ph = (ph or "").strip()
            if ph in ("Succeeded", "Failed"):
                break
            time.sleep(3)
        if ph == "Succeeded":
            return (True, f"wrote {sent}")
        if ph == "Failed":
            _, ev, _ = krun(
                [
                    "-n",
                    ns,
                    "get",
                    "events",
                    "--field-selector",
                    f"involvedObject.name={pod}",
                    "-o",
                    "jsonpath={.items[*].message}",
                ],
                timeout=20,
            )
            code, reason = _mc.classify_mounter_failure(ph, ev or "")
            return (False, f"{code}: {reason}")
        return (
            False,
            f"stamp pod did not finish within the budget (phase {ph or '?'})",
        )
    finally:
        krun(["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--now"], timeout=30)


# ---------------------------------------------------------------------------
# PVC free-space math (pure — testable without a cluster)
# ---------------------------------------------------------------------------


def pvc_space_math(
    free_gib: float | None,
    selected_sizes_gib: list[int | None],
) -> tuple[float | None, bool]:
    """Return (projected_free_gib, will_fit) after selecting the given models.

    If free_gib or any size is None, returns (None, True) — we can't check so we don't block.
    """
    if free_gib is None:
        return None, True
    known_sizes = [s for s in selected_sizes_gib if s is not None]
    if not known_sizes:
        return free_gib, True
    total_needed = sum(known_sizes)
    projected = free_gib - total_needed
    return projected, projected >= 0


def pvc_space_verdict(projected_free_gib: float | None, total_needed_gib: float | None) -> str:
    """PURE — classify a space check as 'ok' | 'marginal' | 'block'. A NEGATIVE projected-free is a
    guaranteed ENOSPC: 'block' (hard refuse — no casual "Proceed anyway?" a user can y through into a
    doomed download), UNLESS the shortfall is marginal (≤ a few % of the download, or ≤10 GiB), which
    stays 'marginal' (warn + proceed). This closes the -302 GiB footgun."""
    if projected_free_gib is None or projected_free_gib >= 0:
        return "ok"
    deficit = -projected_free_gib
    tol = max(0.03 * (total_needed_gib or 0), 10)
    return "marginal" if deficit <= tol else "block"


# ---------------------------------------------------------------------------
# Profile state probe (Phase A)
# ---------------------------------------------------------------------------


def probe_cluster_state(prof: dict, krun=default_krun) -> dict:
    """Probe live cluster state for Phase A panel.

    Returns a dict with keys: ns_ok, gpu_nodes_total, gpu_nodes_used,
    pull_secret_ok, hf_secret_ok, pvc_phase, pvc_free_gib.
    All values may be None if the probe fails (dry-run / cluster unreachable).
    """
    ns = prof.get("NAMESPACE", "")
    gpu_product = prof.get("GPU_PRODUCT", "")
    pull_secret = prof.get("IMAGE_PULL_SECRET", "")
    hf_secret = prof.get("HF_SECRET", "")
    pvc = prof.get("MODEL_CACHE_PVC", "")

    state: dict[str, Any] = {
        "ns_ok": None,
        "gpu_nodes_total": None,
        "gpu_nodes_used": None,
        "pull_secret_ok": None,
        "hf_secret_ok": None,
        "pvc_phase": None,
        "pvc_exists": None,
        "pvc_free_gib": None,
    }

    if not ns:
        return state

    # Namespace
    rc, _, _ = krun(["get", "ns", ns])
    state["ns_ok"] = rc == 0

    # GPU nodes
    if gpu_product:
        rc, out, _ = krun(
            [
                "get",
                "nodes",
                "-l",
                f"nvidia.com/gpu.product={gpu_product}",
                "-o",
                "json",
            ]
        )
        if rc == 0 and out:
            try:
                items = json.loads(out).get("items", [])
                state["gpu_nodes_total"] = len(items)
                # Count nodes with at least 1 GPU in use (simplistic: any GPU request)
                rc2, pout, _ = krun(["get", "pods", "-A", "-o", "json"])
                used_nodes: set[str] = set()
                if rc2 == 0 and pout:
                    pods = json.loads(pout).get("items", [])
                    for pod in pods:
                        if pod.get("status", {}).get("phase") in (
                            "Succeeded",
                            "Failed",
                        ):
                            continue
                        nn = pod["spec"].get("nodeName", "")
                        if not nn:
                            continue
                        gpus = sum(
                            int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0) or 0)
                            for c in (pod["spec"].get("containers", []) + pod["spec"].get("initContainers", []))
                        )
                        if gpus > 0:
                            used_nodes.add(nn)
                state["gpu_nodes_used"] = len(used_nodes)
            except Exception:
                pass

    # Secrets
    if pull_secret:
        rc, _, _ = krun(["-n", ns, "get", "secret", pull_secret])
        state["pull_secret_ok"] = rc == 0
    if hf_secret:
        rc, _, _ = krun(["-n", ns, "get", "secret", hf_secret])
        state["hf_secret_ok"] = rc == 0

    # PVC phase + existence (existence distinguishes "absent → create" from "Pending → leave alone")
    if pvc:
        rc, phase, _ = krun(["-n", ns, "get", "pvc", pvc, "-o", "jsonpath={.status.phase}"])
        state["pvc_exists"] = rc == 0
        if rc == 0 and phase:
            state["pvc_phase"] = phase.strip()

    return state


# ---------------------------------------------------------------------------
# Secret creation helper (Phase A)
# ---------------------------------------------------------------------------


def _create_secret(ns: str, secret_name: str, secret_type: str, value: str, krun=default_krun) -> tuple[int, str]:
    """Apply a k8s Secret from an already-resolved value. Vanilla `kubectl create secret`; no shell, so the
    literal `$oauthtoken` NGC username needs no escaping. Returns (rc, stderr). Shared by the interactive and
    headless ensure paths so a secret is created byte-identically however the value arrived.
    """
    if secret_type == "hf-token":
        rc, _, err = krun(
            [
                "-n",
                ns,
                "create",
                "secret",
                "generic",
                secret_name,
                f"--from-literal=token={value}",
            ]
        )
    else:
        # Docker registry / NGC API key.
        rc, _, err = krun(
            [
                "-n",
                ns,
                "create",
                "secret",
                "docker-registry",
                secret_name,
                "--docker-server=nvcr.io",
                "--docker-username=$oauthtoken",  # NGC requires the literal $oauthtoken (no shell here → no escaping)
                f"--docker-password={value}",
            ]
        )
    return rc, err


def create_secret_interactive(
    ns: str,
    secret_name: str,
    secret_type: str,
    prompt: str,
    krun=default_krun,
    dry_run: bool = False,
) -> bool:
    """Prompt for a secret value and create a k8s Secret in namespace. Returns True on success."""
    import getpass

    print(f"  – {secret_name}: not found")
    value = getpass.getpass(f"    ? {prompt}: ")
    if not value.strip():
        print(f"    ✗ empty value — skipping {secret_name}")
        return False
    if dry_run:
        print(f"    [dry-run] would create secret {secret_name} in {ns}")
        return True
    rc, err = _create_secret(ns, secret_name, secret_type, value, krun=krun)
    if rc == 0:
        print("    → created ✓")
        return True
    else:
        print(f"    ✗ kubectl error: {err.strip()[:120]}")
        return False


# ---------------------------------------------------------------------------
# G9/G10 — "ensure prerequisites": idempotent namespace + secrets on EVERY path
# ---------------------------------------------------------------------------
# The copy/paste (headless) flow used to skip namespace creation entirely (only probed) and skipped secret
# creation unless interactive — so a fresh cluster died on a later preflight/deploy with a manual `kubectl
# create namespace` / `kubectl create secret` step. These fold both into `install` as vanilla, idempotent,
# portable steps that run headless too. Secret VALUES are never baked in code/recipes: they come from a
# well-known env var, the (gitignored) profile, or ~/.config/llmb/secrets — never a committed file.

# The default place an operator can drop token values without exporting them each shell (KEY=VALUE lines,
# same grammar as a profile .env). Kept out of the repo; 0600 recommended.
SECRETS_FILE = Path.home() / ".config" / "llmb" / "secrets"

# Per secret kind: which env vars / profile keys hold the raw value, and a human label. Order = precedence.
_SECRET_KINDS: dict[str, dict] = {
    "hf-token": {
        "env": ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
        "profile": ("HF_TOKEN",),
        "label": "HuggingFace token",
    },
    "ngc-key": {
        "env": ("NGC_API_KEY", "NGC_CLI_API_KEY"),
        "profile": ("NGC_API_KEY",),
        "label": "NGC API key (image pull)",
    },
}


def _read_secrets_file(path: Path = SECRETS_FILE) -> dict[str, str]:
    """Parse a KEY=VALUE secrets file (best-effort; missing file → {}). Same minimal grammar as a profile."""
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text()
    except (FileNotFoundError, OSError):
        return out
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        v = v.strip()
        if v[:1] in ('"', "'") and v[-1:] == v[:1] and len(v) >= 2:
            v = v[1:-1]
        out[k.strip()] = v
    return out


# The user's OWN standard local credential files — the ones huggingface-cli / ngc-cli already write. Reading
# these (never a committed file) is the approved way to make the headless flow zero-paste: the operator
# already logged in with `huggingface-cli login` / `ngc config set`, so their token is on disk. Gitignored by
# nature (they live in $HOME, not the repo).
HF_CRED_FILE = Path.home() / ".cache" / "huggingface" / "token"
NGC_CRED_FILE = Path.home() / ".ngc" / "config"


def _read_hf_token_file(path: Path) -> str | None:
    """`~/.cache/huggingface/token` is a plain file holding just the token (what `huggingface-cli login`
    writes). Best-effort; missing/empty → None."""
    try:
        t = Path(path).read_text().strip()
    except (FileNotFoundError, OSError):
        return None
    return t or None


def _read_ngc_config_apikey(path: Path) -> str | None:
    """`~/.ngc/config` is an INI file (`ngc config set` writes it); the API key is the `apikey = <value>` line
    under `[CURRENT]`. Best-effort; missing/no-key → None. Ignores a literal `no-apikey` placeholder.
    """
    try:
        text = Path(path).read_text()
    except (FileNotFoundError, OSError):
        return None
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith("[") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        if k.strip().lower() == "apikey":
            v = v.strip()
            if v and v.lower() != "no-apikey":
                return v
    return None


# Per secret kind: the user's standard local cred file + its reader. Precedence sits between env and profile,
# so `$HF_TOKEN → ~/.cache/huggingface/token → profile key` (and `$NGC_API_KEY → ~/.ngc/config → profile key`).
_CRED_FILES: dict[str, tuple] = {
    "hf-token": (HF_CRED_FILE, _read_hf_token_file),
    "ngc-key": (NGC_CRED_FILE, _read_ngc_config_apikey),
}


def _resolve_cred_file(secret_type: str, path_overrides: dict | None = None):
    """Resolve a secret VALUE from the user's standard local cred file. `path_overrides` (tests) may point a
    kind at a different path. Returns (value, human-source) or None."""
    spec = _CRED_FILES.get(secret_type)
    if not spec:
        return None
    default_path, reader = spec
    path = (path_overrides or {}).get(secret_type, default_path)
    v = reader(path)
    if v and v.strip():
        return (v.strip(), f"{path}")
    return None


def resolve_secret_value(
    secret_type: str,
    prof: dict,
    environ: dict | None = None,
    secrets_file: Path = SECRETS_FILE,
    cred_files: dict | None = None,
) -> tuple[str | None, str | None]:
    """PURE (env/file injected). Resolve a secret's raw VALUE from the well-known sources, in precedence:
    process env var → the user's standard local cred file (~/.cache/huggingface/token · ~/.ngc/config) →
    profile key → ~/.config/llmb/secrets. Returns (value, human-source) or (None, None) when no source holds
    it. Never reads a committed file and never returns a baked default. `cred_files` overrides the cred-file
    paths for tests."""
    environ = os.environ if environ is None else environ
    spec = _SECRET_KINDS.get(secret_type)
    if not spec:
        return (None, None)
    for name in spec["env"]:
        v = (environ.get(name) or "").strip()
        if v:
            return (v, f"env ${name}")
    cred = _resolve_cred_file(secret_type, cred_files)
    if cred:
        return cred
    for name in spec["profile"]:
        v = (str(prof.get(name) or "")).strip()
        if v:
            return (v, f"profile {name}")
    filevals = _read_secrets_file(secrets_file)
    for name in (*spec["env"], *spec["profile"]):
        v = (filevals.get(name) or "").strip()
        if v:
            return (v, f"{secrets_file}:{name}")
    return (None, None)


def secret_source_hint(secret_type: str, cluster: str) -> str:
    """The EXACT copy/paste remediation when no value source is found — names the precise env var / profile
    key / file, so a missing secret is a one-line fix, never a mid-deploy mystery."""
    spec = _SECRET_KINDS.get(secret_type, {})
    env0 = (spec.get("env") or ("TOKEN",))[0]
    prof0 = (spec.get("profile") or (env0,))[0]
    cred = _CRED_FILES.get(secret_type)
    cred_hint = f" or log in so it lands in {cred[0]}" if cred else ""
    return (
        f"export {env0}=<value>{cred_hint}   (or add {prof0}=<value> to cluster-profiles/{cluster}.env, "
        f"or {env0}=<value> to {SECRETS_FILE})"
    )


def ensure_namespace(
    ns: str,
    krun=default_krun,
    plan_only: bool = False,
    exists: bool | None = None,
    probe: bool = True,
) -> tuple[str, str]:
    """G9 — idempotent namespace ensure (vanilla kubectl, portable). Returns (status, message) where status ∈
    {present, created, planned, failed, skipped}. Safe to re-run: an existing namespace is left untouched; a
    lost create-race (AlreadyExists) resolves to present. `exists` may be injected from an earlier probe;
    `probe=False` (the OFFLINE dry-run) never touches the cluster — it reports a plan with unknown state.
    """
    if not ns:
        return ("skipped", "no NAMESPACE in profile — nothing to ensure")
    if exists is None and probe:
        rc, _, _ = krun(["get", "ns", ns])
        exists = rc == 0
    if exists:
        return ("present", f"namespace '{ns}' already exists")
    if plan_only:
        if exists is None:
            return (
                "planned",
                f"would ensure namespace '{ns}' (offline plan — live state unknown)",
            )
        return ("planned", f"would create namespace '{ns}'")
    rc, _, err = krun(["create", "namespace", ns])
    if rc == 0:
        return ("created", f"created namespace '{ns}'")
    if "AlreadyExists" in (err or ""):
        return ("present", f"namespace '{ns}' already exists (create race)")
    return ("failed", f"could not create namespace '{ns}': {(err or '').strip()[:140]}")


# ---------------------------------------------------------------------------
# Per-recipe model-cache PVCs
#
# Each selected recipe declares its own cache needs (envelope.requires.cache: {size, access_mode, name?}).
# install auto-derives a full PVC spec {name, size, storageClass, accessMode} — with NO operator prompt —
# and idempotently ensures ONE claim per recipe. Two recipes that share a model derive the same namespaced
# name and therefore SHARE one cache (dedup). Unmigrated cells (no requires.cache) fall back to the profile's
# single MODEL_CACHE_* (back-compat), so nothing breaks mid-migration.
# ---------------------------------------------------------------------------

_ACCESS_MODE_MAP = {"rwo": "ReadWriteOnce", "rwx": "ReadWriteMany"}


_SHARED_FS_HINTS = (
    "fsx",
    "lustre",
    "efs",
    "filestore",
    "enterprise-file",
    "nfs",
    "cephfs",
)


def default_pvc_access_mode(storage_class: str) -> str:
    """RWX for a shared/parallel filesystem class (FSx-for-Lustre, EFS, Filestore, DGX Cloud enterprise-file,
    NFS/CephFS — a filesystem many pods mount), else RWO. Last-resort default when a recipe declares no
    access_mode, the profile sets no MODEL_CACHE_ACCESS_MODE, AND no MODEL_CACHE_RWX_CLASS is present. Pure.
    NOTE: the AWS FSx CSI class on these clusters is named `fsx-lustre` (no 'fsx' substring), so the
    hint set matches the class NAME, not just the provisioner."""
    low = (storage_class or "").lower()
    return "ReadWriteMany" if any(h in low for h in _SHARED_FS_HINTS) else "ReadWriteOnce"


def _cache_model_slug(cell: dict) -> str:
    """RFC-1123-safe slug of a cell's model, used for the smart namespaced default cache name. Pure."""
    s = re.sub(r"[^a-z0-9-]+", "-", (cell.get("model") or "").lower()).strip("-")
    return s or "model"


def cache_requires_shared_access(cell: dict) -> bool:
    """PURE — disaggregated serving has independently scheduled cache readers on multiple nodes."""
    return str(cell.get("serving_mode") or "").strip().lower() == "disaggregated"


# ---------------------------------------------------------------------------
# Model-cache claim resolution
# ---------------------------------------------------------------------------
# The cluster profile owns claim names; recipes declare cache size and access mode. All workflow stages use
# model_cache.py so downloads and server mounts resolve the same claim. These names are re-exported for
# compatibility with existing callers.
from model_cache import (  # noqa: E402
    cache_claim_fix_hint,
    model_cache_env_key,
    resolve_cache_claim,
)


def validate_cache_config(cells: list[dict], prof: dict, cluster: str) -> list[str]:
    """Return fatal cache-configuration errors before any download or cluster mutation.

    Every selected cell must resolve to a non-empty claim, and recipes may not declare cluster-specific
    claim names. An empty list means the configuration is valid.
    """
    errs: list[str] = []
    unresolved: list[dict] = []
    for cell in cells:
        name, _src = resolve_cache_claim(cell, prof)
        if not name:
            unresolved.append(cell)
        # `requires.cache.name` is no longer honoured anywhere. A cell still carrying one would have been
        # downloaded into a claim the server never mounts, so refuse rather than silently ignore it.
        declared = ((((cell.get("requires") or {}).get("cache")) or {}).get("name") or "").strip()
        if declared:
            errs.append(
                f"{cell_id(cell)}: recipe declares requires.cache.name='{declared}', which is NOT supported "
                f"— a PVC name is cluster truth, and the rendered manifests mount ${{MODEL_CACHE_PVC}}. "
                f"Remove it from recipe.yaml and {cache_claim_fix_hint(cell, cluster)}."
            )
    # A typo'd per-model key (MODEL_CACHE_PVC_..._NVFP for ..._NVFP4) is otherwise a SILENT no-op: the model
    # quietly falls back to the global default, which is exactly the routing mistake this system exists to
    # prevent. Any MODEL_CACHE_PVC_* key matching no selected model is surfaced as a warning-shaped error.
    known = {model_cache_env_key(c.get("model") or "") for c in cells}
    known.discard("")
    for k in sorted(prof):
        if k.startswith("MODEL_CACHE_PVC_") and k not in known and (prof.get(k) or "").strip():
            errs.append(
                f"profile key {k}='{prof[k]}' matches NO selected recipe's model — it is a silent "
                f"no-op (that model would fall back to MODEL_CACHE_PVC). Expected one of: "
                f"{', '.join(sorted(known)) or '(none)'}"
            )
    if unresolved:
        by_model: dict[str, list[str]] = {}
        for c in unresolved:
            by_model.setdefault(c.get("model") or "?", []).append(cell_id(c))
        for model, ids in sorted(by_model.items()):
            key = model_cache_env_key(model)
            errs.append(
                f"model '{model}' ({len(ids)} cell(s), e.g. {ids[0]}): no model-cache PVC is configured. "
                f"Every rendered manifest mounts ${{MODEL_CACHE_PVC}}, so install cannot download weights "
                f"anywhere the server would find them. Set MODEL_CACHE_PVC "
                f"(or {key}) in cluster-profiles/{cluster}.env."
            )
    return errs


def derive_recipe_cache(
    cell: dict,
    prof: dict,
    model_sizes_gib: list[int | None] | None = None,
    root: Path = ROOT,
) -> dict:
    """PURE — derive the FULL per-recipe model-cache PVC spec from a catalog cell + the resolved profile.
    Never prompts, never touches a cluster. Returns {name, size, size_gib, size_source, size_warning,
    storage_class, access_mode, source} where source ∈ {recipe, profile, derived} names WHY the claim
    looks the way it does (for the plan line).

    Precedence (each field independently):
      access_mode  recipe.requires.cache.access_mode (rwo|rwx) → else MODEL_CACHE_ACCESS_MODE → else a
                   SIZE-AWARE default: a LARGE model (model_size_gib ≥ _FAST_CACHE_MODEL_GIB) → RWX on the fast
                   shared FS (MODEL_CACHE_RWX_CLASS, FSx/Lustre); a SMALL/unknown model → RWO on the block class
                   (MODEL_CACHE_RWO_CLASS, EBS) — big weights load fast off a parallel FS, tiny ones aren't worth
                   FSx-Lustre's large min-filesystem + slow provision. (A legacy MODEL_CACHE_STORAGE_CLASS pin is
                   honored verbatim ABOVE this.)
      storage_class the profile's RWX-or-RWO candidate for the resolved access_mode
                   (MODEL_CACHE_RWX_CLASS / MODEL_CACHE_RWO_CLASS) → else the legacy MODEL_CACHE_STORAGE_CLASS
                   → else '' (cluster-default StorageClass).
      name         explicit cache.name → else the profile's MODEL_CACHE_PVC when set (so install provisions +
                   downloads into the SAME claim the rendered server/bench manifests mount — `${MODEL_CACHE_PVC}`)
                   → else the smart default <model-slug>-model-cache (only when the profile sets NO
                   MODEL_CACHE_PVC). Namespace-scoped, so no collision. NOTE: migration status affects SIZE +
                   storage-class, NOT the name — the name must agree with the mount to avoid a model-not-found.
      size         AUTO-SIZED to fit the model(s) that land in this cache — never a blind floor for a 400GB
                   model (the bug this fixes). = MAX of: recipe.requires.cache.size, profile MODEL_CACHE_SIZE,
                   and auto-fit (SUM of the sharing models' model_size_gib + headroom). An explicit
                   declaration is honored even if SMALLER than the floor (author knows the model is tiny) but
                   is still floored up to the model-fit (never undersize the weights). UNKNOWN model size +
                   nothing declared → 100Gi floor + a size_warning (never a silent undersize).

    `model_sizes_gib` is the list of on-disk GiB for EVERY model that shares this derived cache (so a
    dedup-shared cache is sized to fit them all). When None, it's resolved from THIS cell's recipe.yaml.
    """
    cache = ((cell.get("requires") or {}).get("cache")) or {}
    migrated = bool(cache)

    # Resolve the on-disk size(s) of the model(s) in this cache up-front — it drives BOTH the SIZE-AWARE class
    # default below AND the auto-sizing further down (shared implementation: model_size_gib).
    if model_sizes_gib is None:
        model_sizes_gib = [_cell_model_size_gib(cell, root)]
    _known = [s for s in model_sizes_gib if s and s > 0]
    cache_model_gib = max(_known) if _known else None  # largest model sharing this cache; None if all unknown

    am_raw = (cache.get("access_mode") or "").strip().lower()
    access_mode = _ACCESS_MODE_MAP.get(am_raw, "")  # explicit recipe declaration, or "" (defer to profile)
    rwx_class = (prof.get("MODEL_CACHE_RWX_CLASS") or "").strip()
    rwo_class = (prof.get("MODEL_CACHE_RWO_CLASS") or "").strip()
    legacy_class = (prof.get("MODEL_CACHE_STORAGE_CLASS") or "").strip()

    if cache_requires_shared_access(cell):
        # Topology is stronger than a profile default: frontend/model reconciliation plus independently
        # scheduled prefill/decode workers all read one claim. RWO can attach on only one node, so deriving
        # it here creates a deployment that is structurally incapable of starting. Existing claims are
        # checked against this required mode by ensure_recipe_cache_pvc below.
        access_mode = "ReadWriteMany"
        sc = rwx_class or legacy_class
    elif access_mode:
        # Recipe explicitly declared rwo|rwx → the matching profile class (legacy class as last-resort).
        sc = (rwx_class if access_mode == "ReadWriteMany" else rwo_class) or legacy_class
    else:
        # No recipe declaration → profile default. Precedence:
        #   1. explicit MODEL_CACHE_ACCESS_MODE (operator forced a mode) → its matching class.
        #   2. a back-compat legacy MODEL_CACHE_STORAGE_CLASS pin → honor it VERBATIM (never silently move an
        #      existing cluster's cache to a different class).
        #   3. SIZE-AWARE default: a LARGE model (≥ _FAST_CACHE_MODEL_GIB) → the FAST shared FS (RWX FSx/Lustre)
        #      — it cold-loads far faster than EBS block (~200 MB/s) and RWX lets server + bench co-mount without
        #      the RWO Multi-Attach colocation constraint. A SMALL/unknown model → the RWO block class (EBS):
        #      it provisions + loads in seconds, so FSx-Lustre's large min-filesystem + slow provision is waste.
        prof_mode = (prof.get("MODEL_CACHE_ACCESS_MODE") or "").strip().lower()
        is_large = cache_model_gib is not None and cache_model_gib >= _FAST_CACHE_MODEL_GIB
        if prof_mode in _ACCESS_MODE_MAP:
            access_mode = _ACCESS_MODE_MAP[prof_mode]
            sc = (rwx_class if access_mode == "ReadWriteMany" else rwo_class) or legacy_class
        elif legacy_class:
            sc = legacy_class
            access_mode = default_pvc_access_mode(legacy_class)
        elif is_large and rwx_class:
            access_mode = "ReadWriteMany"
            sc = rwx_class
        elif rwo_class:
            sc = rwo_class
            access_mode = default_pvc_access_mode(rwo_class)
        elif rwx_class:
            # Only a shared FS is configured (no RWO block class) → use it regardless of size.
            access_mode = "ReadWriteMany"
            sc = rwx_class
        else:
            sc = ""
            access_mode = default_pvc_access_mode("")

    # NAME — delegated ENTIRELY to resolve_cache_claim, the one definition every consumer shares (the
    # download Job, deploy.sh's envsubst, sweep.sh, stage-*.sh, preflight, model_load_gate). It reads the
    # PROFILE only, so the claim install writes into and the claim the server mounts are the same string by
    # construction. A recipe's `requires.cache` still supplies SIZE + ACCESS MODE below; it can no longer
    # name a claim (validate_cache_config + schema/envelope.yaml reject `requires.cache.name`).
    # '' means UNRESOLVED — callers must refuse, never invent a name (an invented name is an orphan PVC plus
    # a model-not-found at serve time, which is exactly the failure this replaced).
    name, name_source = resolve_cache_claim(cell, prof)

    # --- SIZE — auto-size to FIT the model(s) (shared implementation: model_size_gib; resolved above). ---
    declared_gib = _parse_quantity_gib(cache.get("size") or "")
    profile_gib = _parse_quantity_gib(prof.get("MODEL_CACHE_SIZE") or "")
    fit_gib = _auto_cache_size_gib(model_sizes_gib)  # model(s) + headroom, or None if unknown
    size_warning: str | None = None

    if declared_gib is not None:
        # Author declared a size → honor it, but NEVER below the model-fit (a too-small declaration can't
        # undersize the weights).
        size_gib = max(declared_gib, fit_gib or 0)
        size_source = (
            "recipe requires.cache.size" if size_gib == declared_gib else "auto-fit (model+headroom > declared size)"
        )
    elif profile_gib is not None:
        size_gib = max(profile_gib, fit_gib or 0)
        size_source = (
            "profile MODEL_CACHE_SIZE" if size_gib == profile_gib else "auto-fit (model+headroom > MODEL_CACHE_SIZE)"
        )
    elif fit_gib is not None:
        size_gib = max(fit_gib, _CACHE_FLOOR_GIB)
        size_source = "auto-fit (model+headroom)"
    else:
        # UNKNOWN model size AND nothing declared → floor, but WARN loudly (never a silent undersize).
        size_gib = _CACHE_FLOOR_GIB
        size_source = "floor (model size unknown)"
        size_warning = (
            f"cache for '{cell_id(cell)}': could not auto-derive size (model not in the known-size "
            f"table and no requires.cache.size / MODEL_CACHE_SIZE) — falling back to the "
            f"{_CACHE_FLOOR_GIB}Gi floor. If the model is larger, set requires.cache.size in "
            f"recipe.yaml (or MODEL_CACHE_SIZE in the profile) to avoid an out-of-space download."
        )
    size = _fmt_gib(size_gib)

    # `source` names WHY the claim looks the way it does, for the plan line. The NAME's provenance is now
    # always a profile key (or 'unresolved'); `migrated` only tells you the SHAPE came from the recipe.
    source = name_source or "unresolved"
    return {
        "name": name,
        "name_source": source,
        "size": size,
        "size_gib": size_gib,
        "size_source": size_source,
        "size_warning": size_warning,
        "storage_class": sc,
        "access_mode": access_mode,
        "source": ("recipe" if migrated else source),
    }


# Object-store provisioners can't back a block/file PVC — a model-cache claim on one sits Pending FOREVER
# (the QA hang). Fail fast with guidance instead of applying a PVC that never binds. (Same taxonomy as
# cluster_readiness._OBJECT_STORE_PROVISIONERS.)
_OBJECT_STORE_PROVISIONERS = ("s3.csi", "blob.csi", "gcs.csi", "objectstorage", "cosi.")


def is_object_store_provisioner(provisioner: str) -> bool:
    """PURE. True iff `provisioner` is an object/bucket CSI that cannot dynamically provision a block/file
    PVC. Unit-testable with a plain string."""
    prov = (provisioner or "").lower()
    return bool(prov) and any(o in prov for o in _OBJECT_STORE_PROVISIONERS)


def storageclass_provisioner(krun, sc: str) -> str:
    """Return a StorageClass's provisioner (lowercased), or '' when unset/unreadable (safe-degrade → the
    caller proceeds rather than a false fail)."""
    if not sc:
        return ""
    rc, out, _ = krun(["get", "storageclass", sc, "-o", "jsonpath={.provisioner}"])
    return (out or "").strip().lower() if rc == 0 else ""


def render_model_cache_pvc_manifest(ns: str, name: str, size: str, storage_class: str, access_mode: str) -> str:
    """PURE — the PVC YAML `install` applies for a per-recipe model-cache claim. Labeled managed=true; an
    empty storage_class falls through to the cluster default (line omitted). No recipe field flows into a
    RENDERED serving/bench manifest, so this has ZERO recipe_hash impact."""
    sc_line = f"  storageClassName: {storage_class}\n" if storage_class else ""
    return (
        "apiVersion: v1\n"
        "kind: PersistentVolumeClaim\n"
        "metadata:\n"
        f"  name: {name}\n"
        f"  namespace: {ns}\n"
        "  labels:\n"
        '    llmb.nvidia.com/managed: "true"\n'
        "spec:\n"
        "  accessModes:\n"
        f"    - {access_mode}\n"
        f"{sc_line}"
        "  resources:\n"
        "    requests:\n"
        f"      storage: {size}\n"
    )


def _shared_cache_names() -> tuple[str, ...]:
    """Well-known shared model-cache PVC names, env-overridable (LLMB_SHARED_CACHE_NAMES, comma-separated)."""
    override = (os.environ.get("LLMB_SHARED_CACHE_NAMES") or "").strip()
    if override:
        return tuple(n.strip() for n in override.split(",") if n.strip())
    return _SHARED_CACHE_NAMES


def _shared_cache_label() -> str:
    """The label whose PRESENCE marks a shared model cache, env-overridable (LLMB_SHARED_CACHE_LABEL)."""
    return (os.environ.get("LLMB_SHARED_CACHE_LABEL") or "").strip() or _SHARED_CACHE_LABEL


# pick_shared_cache() / discover_shared_model_cache() were DELETED here (T10).
#
# They existed to auto-adopt a pre-existing shared cache by writing prof["MODEL_CACHE_PVC"] in memory. That
# value was visible to install and to nothing else — deploy.sh, sweep.sh, stage-*.sh and preflight all
# re-read the profile FILE — so it moved the weights somewhere the server never mounted. Keeping a
# name-heuristic "which cache should we use?" helper around is a trap for the next person who wants
# discovery to decide something. list_cache_candidates() below is the honest replacement: it LISTS what
# exists, with no size floor and an explicit probe_error, and the operator (or --adopt-cache) writes the
# answer into the profile FILE where every consumer reads it.


def list_cache_candidates(ns: str, krun=default_krun) -> tuple[list[dict], str]:
    """Every PVC in `ns` that could plausibly be a model cache, NEWEST-STYLE: no size floor.

    The ≥1Ti floor in pick_shared_cache is a RANKING hint for auto-selection. As a VISIBILITY gate it hid
    this cluster's dedicated 800Gi `nemotron-ultra-nvfp4-cache` — which held a complete copy of the weights
    — so every model was routed into the 1200Gi GLM-5 claim and the download died at
    'OSError [Errno 122] Disk quota exceeded'. When we only SUGGEST, there is no reason to hide anything:
    show them all and let the operator pick.

    Returns (candidates, probe_error). ABSENCE IS NOT ZERO: a failed listing returns ([], "<why>") and the
    caller MUST render that as "could not look", never as "none found" — reporting a probe failure as an
    empty inventory is the same defect class as everything else in this file. Sorted largest-first.
    """
    if not ns:
        return ([], "no NAMESPACE in profile")
    rc, out, err = krun(["-n", ns, "get", "pvc", "-o", "json"])
    if rc != 0:
        return (
            [],
            f"kubectl get pvc failed: {(err or '').strip()[:120] or f'rc={rc}'}",
        )
    if not out.strip():
        return ([], "kubectl get pvc returned no output")
    try:
        items = (json.loads(out) or {}).get("items") or []
    except Exception as e:
        return ([], f"unparseable kubectl JSON: {e}")
    cands: list[dict] = []
    for it in items:
        meta, spec, status = (
            it.get("metadata") or {},
            it.get("spec") or {},
            it.get("status") or {},
        )
        nm = meta.get("name") or ""
        low = nm.lower()
        labels = meta.get("labels") or {}
        looks_like_cache = (
            nm in _shared_cache_names()
            or _shared_cache_label() in labels
            or "llmb.nvidia.com/model-name" in labels
            or any(h in low for h in _CACHE_CANDIDATE_HINTS)
        )
        # `-artifacts` claims are per-cell run output, never weights — excluding them keeps the advice
        # readable on a namespace with dozens of them (this cluster has 25).
        if not looks_like_cache or low.endswith("-artifacts"):
            continue
        cap_raw = (
            (status.get("capacity") or {}).get("storage")
            or (spec.get("resources") or {}).get("requests", {}).get("storage")
            or ""
        )
        cands.append(
            {
                "name": nm,
                "phase": status.get("phase") or "",
                "capacity_gib": _parse_quantity_gib(cap_raw),
                "access_modes": spec.get("accessModes") or [],
                "labels": labels,
            }
        )
    return (sorted(cands, key=lambda c: (-(c.get("capacity_gib") or 0), c["name"])), "")


def render_cache_candidates_advice(cands: list[dict], probe_error: str, cells: list[dict], cluster: str) -> str:
    """PURE — the operator-facing block printed when a claim is UNRESOLVED. It prints the exact profile
    lines to paste AND the one-command way to have install write them (`--adopt-cache`), then install stops.
    It never silently edits the profile: adoption is an explicit operator act, because the profile file is
    the one place every consumer reads and a value only install knows is a divergence by construction.
    """
    out = ["  Candidate model-cache PVCs (read-only listing; install will NOT choose for you):"]
    if probe_error:
        # ABSENCE-AS-SIGNAL: we could not look. Saying "none found" here would tell the operator to create a
        # cache that may already exist — the same lie as reporting a failed probe as zero.
        out.append(f"    UNKNOWN — could not list PVCs: {probe_error}")
        out.append("    (this is 'we could not look', NOT 'there are none' — fix cluster access and re-run)")
    elif not cands:
        out.append(
            "    (none — `kubectl get pvc` succeeded and showed no claim that looks like a model "
            "cache; you will need to create one)"
        )
    for c in cands:
        cap = f"{c['capacity_gib']}Gi" if c.get("capacity_gib") else "?"
        out.append(
            f"    {c['name']:<44} {cap:>8}  {','.join(c.get('access_modes') or []) or '?':<14} "
            f"{c.get('phase') or '?'}"
        )
    models = sorted({(c.get("model") or "?") for c in cells})
    out.append(f"\n  Add to cluster-profiles/{cluster}.env — the default claim for every model:")
    out.append('    MODEL_CACHE_PVC="<pvc-from-the-list-above>"')
    if len(models) > 1:
        out.append("  ...and, for any model whose weights live in a DIFFERENT claim, a per-model override:")
        for m in models:
            out.append(f'    {model_cache_env_key(m)}="<pvc>"   # model {m}')
    if cands:
        out.append("\n  Or have install write them for you (it picks the best-fitting claim per model,")
        out.append("  prints the diff, and applies it to the profile FILE — the one place every consumer reads):")
        out.append(f"    llmb-k8s install {cluster} --adopt-cache")
    return "\n".join(out)


def plan_cache_adoption(
    cells: list[dict], cands: list[dict], prof: dict, root: Path = ROOT
) -> tuple[dict[str, str], list[str]]:
    """Choose and report a suitable existing claim for each model without an explicit mapping.

    Selection prefers a claim already stamped for the model, then the smallest candidate with sufficient
    capacity. Explicit per-model mappings are never overwritten, and each decision or skip is returned in
    `notes`. The resulting keys are written to the cluster profile by `--adopt-cache`.
    """
    chosen: dict[str, str] = {}
    notes: list[str] = []
    by_model: dict[str, list[dict]] = {}
    for c in cells:
        by_model.setdefault(c.get("model") or "", []).append(c)
    default_claim = (prof.get("MODEL_CACHE_PVC") or "").strip()
    for model, group in sorted(by_model.items()):
        if not model:
            continue
        key = model_cache_env_key(model)
        if (prof.get(key) or "").strip():
            notes.append(f"model '{model}': already pinned by {key}='{prof[key]}' — left alone")
            continue  # explicit per-model key — never overwrite

        # The PVC stamp is written by model-download.yaml.j2, whose label filter keeps [a-z0-9-_.] while
        # _cache_model_slug() folds '_' and '.' to '-'. Comparing the two directly would silently lose the
        # reuse branch for any model name containing '_' or '.' — and losing it means re-downloading
        # hundreds of GiB that are already on the claim. Compare on a common normalisation instead.
        def _norm_label(s: str) -> str:
            return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")

        slug = _cache_model_slug(group[0])
        slug_n = _norm_label(slug)
        need = _auto_cache_size_gib([_cell_model_size_gib(group[0], root)])
        # "Already holds this model" is read from the PVC stamp the download Job writes on success. Match on
        # ANY of the three label shapes: the per-model key (llmb.nvidia.com/model.<slug>, which accumulates)
        # and the two single-valued legacy ones, which are merge-patched and so record only the LAST model
        # written to a shared cache. Live example: nemotron-ultra-nvfp4-cache carries model-revision=183968f8
        # and NO model-name at all.
        rev = _cell_model_revision(group[0], root)

        def _stamp_match(c: dict, slug_n: str = slug_n, rev: str = rev) -> str:
            lbl = c.get("labels") or {}
            for lk, lv in lbl.items():
                if _mc.model_from_stamp_key(lk) == _norm_label(slug_n):
                    return f"stamped {lk}={lv}"
            if _norm_label(lbl.get("llmb.nvidia.com/model-name") or "") == slug_n:
                return f"stamped model-name={lbl['llmb.nvidia.com/model-name']}"
            lrev = lbl.get("llmb.nvidia.com/model-revision") or ""
            if rev and lrev and rev.startswith(lrev):
                return f"stamped model-revision={lrev[:12]}"
            return ""

        stamped = [(c, w) for c, w in ((c, _stamp_match(c)) for c in cands) if w]
        default_cap = next(
            (c.get("capacity_gib") or 0 for c in cands if c["name"] == default_claim),
            None,
        )
        fits = [c for c in cands if (c.get("capacity_gib") or 0) >= (need or 0)]
        if stamped:
            pick, why = (
                stamped[0][0]["name"],
                f"already holds {model} ({stamped[0][1]})",
            )
        elif default_claim and need is None:
            # UNKNOWN model size and a default already in force. There is no EVIDENCE the default is wrong,
            # and "smallest claim that fits 0 GiB" is a coin flip dressed as a decision. Leave it alone.
            notes.append(
                f"model '{model}': size unknown and MODEL_CACHE_PVC='{default_claim}' already "
                f"covers it — left alone (no evidence to move it)"
            )
            continue
        elif default_claim and default_cap is not None and default_cap >= (need or 0):
            notes.append(
                f"model '{model}': MODEL_CACHE_PVC='{default_claim}' ({default_cap}Gi) already "
                f"fits ~{need} GiB — left alone"
            )
            continue
        elif need is None and cands:
            # UNKNOWN size and no default to inherit. "Smallest that fits 0 GiB" is a coin flip; with no
            # size information the only defensible rule is MAXIMUM HEADROOM. Say which it is and why.
            pick = max(cands, key=lambda c: (c.get("capacity_gib") or 0, c["name"]))["name"]
            why = "model size unknown — largest claim, for maximum headroom"
        elif fits:
            pick = min(fits, key=lambda c: (c.get("capacity_gib") or 0, c["name"]))["name"]
            why = f"smallest claim that fits ~{need} GiB"
        else:
            notes.append(
                f"model '{model}': no candidate claim is large enough "
                f"(~{need if need is not None else '?'} GiB needed) — create one and set "
                f"{key} by hand"
            )
            continue
        if pick == default_claim:
            notes.append(f"model '{model}': best claim IS the current default '{default_claim}' — no change")
            continue
        chosen[key] = pick
        notes.append(f"model '{model}' → {pick}  ({why}; writes {key})")
    return (chosen, notes)


def apply_profile_keys(env_path: Path, keys: dict[str, str]) -> list[str]:
    """Write KEY="value" into a cluster-profile .env, replacing an existing assignment in place and
    appending the rest. Returns the changed lines for the caller to print. Idempotent.

    Deliberately line-oriented and comment-preserving: a cluster profile is hand-edited by operators and a
    rewrite that reflows it would lose the annotations that explain the cluster."""
    text = env_path.read_text() if env_path.exists() else ""
    lines = text.splitlines()
    changed: list[str] = []
    remaining = dict(keys)
    # Replace every assignment so the last value read by the shell cannot override the update.
    seen: set[str] = set()
    for i, ln in enumerate(lines):
        m = re.match(r"^(\s*)([A-Z_][A-Z0-9_]*)=", ln)
        if not m or m.group(2) not in remaining:
            continue
        k = m.group(2)
        lines[i] = f'{k}="{remaining[k]}"'
        if k in seen:
            changed.append(f'{k}="{remaining[k]}"   <- duplicate assignment on line {i + 1}, also rewritten')
        else:
            changed.append(lines[i])
            seen.add(k)
    for k in seen:
        remaining.pop(k, None)
    if remaining:
        lines.append("")
        lines.append("# model-cache claims adopted by `llmb-k8s install --adopt-cache` (cluster truth: every")
        lines.append("# consumer — install, deploy.sh, sweep.sh, stage-*.sh, preflight — reads these).")
        for k, v in sorted(remaining.items()):
            lines.append(f'{k}="{v}"')
            changed.append(f'{k}="{v}"')
    env_path.write_text("\n".join(lines) + "\n")
    return changed


def cache_bind_verdict(phase: str, events_text: str) -> str:
    """PURE — classify a freshly-applied cache PVC → 'bound' | 'failed' | 'pending'. 'failed' on a clear
    provisioning/ATTACH error in the events (the cinder FailedAttachVolume case); 'bound' when phase==Bound;
    else 'pending'. A WaitForFirstConsumer PVC stays Pending until a consumer — that is NOT a failure.
    """
    if (phase or "").strip() == "Bound":
        return "bound"
    low = (events_text or "").lower()
    for sig in (
        "provisioningfailed",
        "failedattachvolume",
        "cannot be attached",
        "could not create volume",
        "failed to provision",
        "attachvolume.attach failed",
    ):
        if sig in low:
            return "failed"
    return "pending"


def wait_for_cache_bind(
    ns: str,
    name: str,
    krun=default_krun,
    budget_s: int = _CACHE_BIND_BUDGET_S,
    poll_s: int = 5,
) -> tuple[str, str]:
    """Poll a freshly-applied cache PVC up to `budget_s`, FAST-failing on a CSI provisioning/attach error so a
    invalid class fails promptly. Returns (verdict, detail) where verdict ∈ {bound, failed, pending}.
    'pending' at timeout is benign (WaitForFirstConsumer binds once a consumer exists). Injected krun in tests.
    """
    deadline = time.time() + max(0, budget_s)
    while True:
        rc, phase, _ = krun(["-n", ns, "get", "pvc", name, "-o", "jsonpath={.status.phase}"])
        rc2, ev, _ = krun(
            [
                "-n",
                ns,
                "get",
                "events",
                "--field-selector",
                f"involvedObject.name={name}",
                "-o",
                "jsonpath={.items[*].message}",
            ]
        )
        verdict = cache_bind_verdict(phase if rc == 0 else "", ev if rc2 == 0 else "")
        if verdict == "bound":
            return ("bound", "Bound")
        if verdict == "failed":
            return ("failed", (ev or "").strip()[:200])
        if time.time() >= deadline:
            return (
                "pending",
                "not bound within budget (WaitForFirstConsumer is normal until a consumer mounts it)",
            )
        time.sleep(max(0, poll_s))


def ensure_recipe_cache_pvc(
    spec: dict,
    ns: str,
    krun=default_krun,
    applier=default_apply,
    plan_only: bool = False,
    exists: bool | None = None,
    probe: bool = True,
) -> tuple[str, str]:
    """Idempotent ensure of ONE per-recipe model-cache PVC (rebuilt + generalized from the kvbm-clean-slate
    G3 fix). Returns (status, message) where status ∈ {present, created, planned, failed, skipped}. NEVER
    clobbers an existing PVC (Bound or otherwise) — a claim holds real weights and PVC specs are largely
    immutable. `probe=False` (OFFLINE dry-run) never touches the cluster."""
    name = (spec.get("name") or "").strip()
    if not name:
        return ("skipped", "no cache name derived — nothing to ensure")
    if not ns:
        return ("skipped", "no NAMESPACE in profile — cannot ensure PVC")
    size = spec.get("size") or "100Gi"
    need_gib = spec.get("size_gib") or _parse_quantity_gib(size) or 0
    sc = spec.get("storage_class") or ""
    access = spec.get("access_mode") or "ReadWriteOnce"
    if exists is None and probe:
        rc, _, _ = krun(["-n", ns, "get", "pvc", name])
        exists = rc == 0
    if exists:
        # An existing claim is immutable cluster truth. Do not call it "present" when its access mode
        # contradicts the topology-derived requirement: install would otherwise proceed to download and
        # the later multi-node deployment would hang in Init with FailedAttachVolume/Multi-Attach.
        if probe and access == "ReadWriteMany":
            rc_mode, modes_out, _ = krun(["-n", ns, "get", "pvc", name, "-o", "jsonpath={.spec.accessModes}"])
            if rc_mode != 0 or not (modes_out or "").strip():
                return (
                    "access-unknown",
                    f"PVC '{name}' exists but its access mode could not be verified; refusing to assume it "
                    f"satisfies required {access} access.",
                )
            actual_modes = set(re.findall(r"Read(?:Write|Only)(?:Once|Many)", modes_out))
            if "ReadWriteMany" not in actual_modes:
                shown = ",".join(sorted(actual_modes)) or modes_out.strip()
                return (
                    "access-mismatch",
                    f"PVC '{name}' is {shown}, but this disaggregated recipe requires a multi-node "
                    "ReadWriteMany cache for model preparation and serving. Point the profile at a populated "
                    "RWX claim or provision one with MODEL_CACHE_RWX_CLASS; the existing claim was left "
                    "untouched.",
                )
        # BELT-AND-SUSPENDERS: a claim that already exists but is SMALLER than this recipe now needs would
        # ENOSPC mid-download (the exact QA footgun). Refuse + guide (resize/recreate) rather than downloading
        # into certain failure. A marginally-smaller PVC (≤5%) is tolerated (headroom is slack).
        if probe and need_gib:
            cap = probe_pvc_capacity_gib(ns, name, krun)
            if cap is not None and cap < need_gib - math.ceil(need_gib * _CACHE_UNDERSIZE_FRAC):
                return (
                    "undersized",
                    f"PVC '{name}' exists at ~{cap}Gi but this recipe needs ≥{need_gib}Gi (model + "
                    f"headroom); a download would run out of space. Resize the claim to ≥{need_gib}Gi "
                    f"(expand it if the StorageClass allows, or delete it if empty and re-run install).",
                )
        return ("present", f"PVC '{name}' already exists (left untouched)")
    if plan_only:
        if exists is None:
            return (
                "planned",
                f"would ensure PVC '{name}' ({size}, {access}, sc={sc or 'cluster-default'}) "
                "(offline plan — live state unknown)",
            )
        return (
            "planned",
            f"would create PVC '{name}' ({size}, {access}, sc={sc or 'cluster-default'})",
        )
    # Fail FAST on an object-store class — the claim would sit Pending forever (never binds). Cheap
    # provisioner lookup before we apply a doomed PVC.
    if probe and sc:
        prov = storageclass_provisioner(krun, sc)
        if is_object_store_provisioner(prov):
            return (
                "failed",
                f"storage class '{sc}' uses object-store provisioner '{prov}' — it cannot "
                "provision a block/file PVC (the claim would sit Pending forever). Set a block/file class "
                "(e.g. ebs / fsx-lustre) in the profile; kubectl get storageclass",
            )
    manifest = render_model_cache_pvc_manifest(ns, name, size, sc, access)
    rc, err = applier(manifest, ns)
    if rc == 0:
        return (
            "created",
            f"created PVC '{name}' ({size}, {access}, sc={sc or 'cluster-default'})",
        )
    if "AlreadyExists" in (err or ""):
        return ("present", f"PVC '{name}' already exists (create race)")
    return ("failed", f"could not create PVC '{name}': {(err or '').strip()[:140]}")


def ensure_recipe_cache_pvc_validated(
    spec: dict,
    ns: str,
    prof: dict,
    krun=default_krun,
    applier=default_apply,
    plan_only: bool = False,
    probe: bool = True,
) -> tuple[str, str, dict]:
    """Ensure ONE cache PVC, then VALIDATE it binds/attaches within a short budget and FAST-FALL-BACK to the
    RWX class if the derived (block) class fails to attach — the shared-cluster block-storage FailedAttachVolume case, which
    otherwise remains pending. Returns (status, message, effective_spec). Statuses add 'fell-back' (recreated on
    the RWX class) and 'attach-failed' (failed with no distinct RWX fallback available). Never validates in
    plan/offline mode."""
    st, msg = ensure_recipe_cache_pvc(spec, ns, krun=krun, applier=applier, plan_only=plan_only, probe=probe)
    if st != "created" or plan_only or not probe:
        return (st, msg, spec)
    verdict, detail = wait_for_cache_bind(ns, spec["name"], krun)
    if verdict != "failed":
        return (st, msg, spec)  # bound, or benign WaitForFirstConsumer pending
    failed_sc = spec.get("storage_class") or "cluster-default"
    rwx_class = (prof.get("MODEL_CACHE_RWX_CLASS") or "").strip()
    if not rwx_class or rwx_class == (spec.get("storage_class") or ""):
        return (
            "attach-failed",
            f"cache PVC '{spec['name']}' storage class '{failed_sc}' failed to attach ({detail}); "
            f"no distinct RWX fallback class is configured — set MODEL_CACHE_RWX_CLASS to a "
            f"shared or multi-attach storage class available on this cluster.",
            spec,
        )
    # The failed PVC never bound/attached → it is empty; delete it and recreate on the RWX (shared FS) class.
    krun(["-n", ns, "delete", "pvc", spec["name"], "--ignore-not-found", "--wait=false"])
    fb = {
        **spec,
        "storage_class": rwx_class,
        "access_mode": "ReadWriteMany",
        "size_source": f"{spec.get('size_source', '')} + fallback (block class '{failed_sc}' failed to attach)",
    }
    st2, msg2 = ensure_recipe_cache_pvc(fb, ns, krun=krun, applier=applier, plan_only=False, probe=False, exists=False)
    return (
        "fell-back",
        f"block class '{failed_sc}' failed to attach ({detail}); fell back to RWX class '{rwx_class}' — {msg2}",
        fb,
    )


def ensure_recipe_cache_pvcs(
    cells: list[dict],
    prof: dict,
    krun=default_krun,
    applier=default_apply,
    plan_only: bool = False,
    probe: bool = True,
) -> list[tuple[str, str, dict]]:
    """PER-RECIPE cache provisioning: for each selected cell, derive its cache spec and idempotently ensure
    the PVC. Dedups by claim name (recipes sharing a model share one cache), so `install --recipes A,B,C`
    creates exactly one CORRECT cache per distinct claim — never a single one-size cache that can't serve
    them all. Prints one line per claim and returns [(status, message, spec)] for tests.

    ROBUSTNESS: a freshly-provisioned PVC is VALIDATED to bind/attach within a short budget, with a
    FAST-FALL-BACK to the RWX class if a broken block class (cinder) fails to attach.

    (This docstring used to open with "DISCOVER a pre-existing shared model-cache PVC … and PREFER it — use
    it (set MODEL_CACHE_PVC) + skip provisioning". That behaviour is GONE: pick_shared_cache and
    discover_shared_model_cache were deleted, and setting prof["MODEL_CACHE_PVC"] in memory was the original
    defect — only install could see it, while deploy.sh/sweep.sh/preflight re-read the profile FILE. The
    claim name now comes from resolve_cache_claim, full stop; discovery survives only as ADVICE the operator
    can act on, or as --adopt-cache writing into the profile file.)"""
    print("\n── Per-recipe model caches ────────────────────────────────────────")
    if not plan_only:
        print(
            _stage_note(
                "Provisioning model-cache PVC(s)",
                "~1-2 min",
                "a WaitForFirstConsumer block class (e.g. EBS) stays Pending until the first pod "
                "mounts it — a brief Pending here is NORMAL, not a hang",
            )
        )
    results: list[tuple[str, str, dict]] = []
    ns = prof.get("NAMESPACE", "")
    # Claim names come only from the profile; discovery is advisory unless explicitly adopted.
    # First pass: derive base specs (name/class/mode), then group model sizes by claim so a cache SHARED by
    # several models is sized to fit them ALL.
    #
    # Deduplicate by model because multiple cells sharing a model use one copy of its weights.
    base = [(cell, derive_recipe_cache(cell, prof)) for cell in cells]
    sizes_by_claim: dict[str, dict[str, int | None]] = {}
    shared_access_by_claim: dict[str, bool] = {}
    for cell, spec0 in base:
        # UNREADABLE (None) falls back to the cell id, which is unique — an unreadable cell is counted on
        # its own rather than merged with other unreadable ones.
        _repo = _cell_model_repo(cell)
        key = _repo if _repo else ((cell.get("model") or cell_id(cell)) if _repo == "" else cell_id(cell))
        sizes_by_claim.setdefault(spec0["name"], {})[key] = _cell_model_size_gib(cell)
        shared_access_by_claim[spec0["name"]] = shared_access_by_claim.get(
            spec0["name"], False
        ) or cache_requires_shared_access(cell)
    sizes_by_name = {n: list(d.values()) for n, d in sizes_by_claim.items()}
    seen: set[str] = set()
    for cell, spec0 in base:
        name = spec0["name"]
        if name in seen:
            print(f"  · {cell_id(cell)} → shares cache '{name}' (already ensured)")
            continue
        seen.add(name)
        # A claim shared by aggregate + disaggregated cells takes the strictest access requirement,
        # independent of catalog ordering. Without this, an aggregate-first selection could provision RWO
        # and the later disaggregated consumer would inherit an unusable claim.
        shape_cell = {**cell, "serving_mode": "disaggregated"} if shared_access_by_claim.get(name) else cell
        spec = derive_recipe_cache(shape_cell, prof, model_sizes_gib=sizes_by_name[name])
        if spec.get("size_warning"):
            print(f"  ⚠ {spec['size_warning']}")
        st, msg, eff_spec = ensure_recipe_cache_pvc_validated(
            spec, ns, prof, krun=krun, applier=applier, plan_only=plan_only, probe=probe
        )
        print(f"  {_ENSURE_GLYPH.get(st, '?')} {cell_id(cell)} [{eff_spec['source']}, {eff_spec['size']}] → {msg}")
        results.append((st, msg, eff_spec))
    if not results:
        print("  (no caches to ensure)")
    return results


def cache_pvc_by_repo(cells: list[dict], prof: dict, root: Path = ROOT) -> tuple[dict[str, str], list[str]]:
    """Map each selected cell's model_repo (normalized) → THE cache PVC its weights belong in, so Phase C
    downloads each model into the claim its server will mount.

    Returns `(map, errors)`. Unreadable recipes or unresolved claims are errors rather than implicit
    fallbacks. The claim is obtained directly from `resolve_cache_claim` so all consumers use one rule.
    """
    out: dict[str, str] = {}
    errors: list[str] = []
    for cell in cells:
        rp = root / cell.get("_path", "") / "recipe.yaml"
        try:
            r = yaml.safe_load(rp.read_text()) or {}
        except Exception as e:
            errors.append(f"{cell_id(cell)}: cannot read {rp} ({e}) — cannot determine its download target")
            continue
        repo = ((r.get("serving") or {}).get("model_repo") or "").strip()
        if not repo:
            continue  # genuinely no model to download (declaration-only cell)
        claim, _src = resolve_cache_claim(cell, prof)
        if not claim:
            errors.append(f"{cell_id(cell)}: {cache_claim_fix_hint(cell, prof.get('CLUSTER', '<cluster>'))}")
            continue
        prev = out.get(_norm_repo(repo))
        if prev and prev != claim:
            errors.append(
                f"model repo {repo} resolves to TWO different claims ({prev} and {claim}) across the "
                f"selected cells — one model's weights cannot live in two places; reconcile "
                f"{model_cache_env_key(cell.get('model') or '')} in the profile"
            )
            continue
        out[_norm_repo(repo)] = claim
    return (out, errors)


def capacity_gate(
    dl_models: list[dict],
    cache_by_repo: dict[str, str],
    ns: str,
    krun=default_krun,
    do_probe: bool = True,
    plan_only: bool = False,
    allow_unmeasured: bool = False,
    prof: dict | None = None,
) -> tuple[list[dict], list[str]]:
    """PER-CLAIM capacity guard, shared by the headless AND interactive doors. Returns (models_to_download,
    blocked_claims) and prints the reasoning.

    Two defects this replaces:
      1. The check compared the WHOLE union against ONE claim's free space (state["pvc_free_gib"], probed
         against the raw profile MODEL_CACHE_PVC). With per-model claims that answers the wrong question,
         so group the models by the claim they actually land in and measure each.
      2. UNKNOWN free space PROCEEDED ("probe unavailable — proceeding") and then wrote until ENOSPC at
         shard 102/113, leaving a partial tree squatting on a claim a live run depended on. Unknown
         capacity does not authorise a large write.

    `allow_unmeasured` is the deliberate escape hatch: on clusters where restricted PSA/Kyverno blocks the
    `kubectl run` mounter pod (restricted clusters), free space is PERMANENTLY unknowable and refusing without
    an override would make large downloads impossible rather than merely careful."""
    prof = prof or {}
    by_claim: dict[str, list[dict]] = {}
    for m in dl_models:
        by_claim.setdefault(cache_by_repo.get(_norm_repo(m.get("model_repo") or "")) or "", []).append(m)
    blocked: list[str] = []
    for claim, ms in sorted(by_claim.items()):
        sizes = [(m, model_size_gib(m)) for m in ms]
        total = sum(s for _, s in sizes if s)
        unknown = [m["model_name"] for m, s in sizes if not s]
        print(
            f"  ── {claim or '(unresolved)'}: ~{total} GiB to download"
            + (f" (+{len(unknown)} of unknown size: {', '.join(unknown[:3])})" if unknown else "")
        )
        # Live planning probes real capacity and reports whether the proposed download fits without applying changes.
        if plan_only and not do_probe:
            continue
        if not claim:
            # No resolved claim = no idea where these weights go. Never download on that.
            print(
                "     ❌ REFUSING: no model-cache claim resolved for these model(s) — the download would "
                "have no destination the server will mount."
            )
            blocked.append(claim)
            continue
        _pw: list = []
        free = probe_pvc_free_gib(ns, claim, krun=krun, prof=prof, why=_pw) if do_probe else None
        if free is None:
            if not do_probe:
                why = "the PVC free-space probe was disabled (--skip-pvc-probe)"
            elif _pw:
                # SAY WHICH FAILURE IT WAS. unschedulable / mount-failed / rbac-denied are three different
                # operator actions and they used to collapse into one silent None that read as "the cache
                # is broken".
                _code, _reason, _hint = _pw[0]
                why = f"{_code}: {_reason}" + (f". {_hint}" if _hint else "")
            else:
                why = (
                    "the free-space probe could not read the claim (it runs a short-lived pod that "
                    "mounts the PVC; restricted PodSecurity/Kyverno or RBAC can block it)"
                )
            if total < _UNMEASURED_DOWNLOAD_MAX_GIB and not unknown:
                print(
                    f"     ⚠  free space on '{claim}' is UNKNOWN — {why} — but the write is small "
                    f"(~{total} GiB < {_UNMEASURED_DOWNLOAD_MAX_GIB} GiB), so proceeding."
                )
            elif allow_unmeasured:
                print(
                    f"     ⚠  free space on '{claim}' is UNKNOWN — {why}. Proceeding anyway because "
                    f"--allow-unmeasured-download was given. If ~{total} GiB does not fit, the download "
                    f"will die partway and leave a partial tree on the claim."
                )
            else:
                print(
                    f"     ❌ REFUSING: free space on '{claim}' is UNKNOWN and this is a ~{total} GiB "
                    f"write. Unknown capacity does not authorise it."
                )
                print(f"        Why unknown: {why}.")
                print(
                    "        Either make the probe work, or accept the risk explicitly with "
                    "--allow-unmeasured-download, or re-run with --skip-model-download."
                )
                blocked.append(claim)
            continue
        print(f"     free on '{claim}': ~{int(free)} GiB")
        if total and pvc_space_verdict(free - total, total) == "block":
            print(
                f"     ❌ {'WOULD NOT FIT' if plan_only else 'REFUSING'}: needs ~{total} GiB, "
                f"only ~{int(free)} GiB free on '{claim}'."
            )
            print(
                "        A partial download does not roll itself back — it squats on the cache and the "
                "next run fails on quota."
            )
            print(
                f"        Narrow the set with `--recipes <cell>[,<cell>]`, free space on '{claim}', or "
                f"point this model at another claim via "
                f"{model_cache_env_key(ms[0].get('model_name') or '')} in the profile."
            )
            blocked.append(claim)
    if blocked and not plan_only:
        dl_models = [
            m for m in dl_models if (cache_by_repo.get(_norm_repo(m.get("model_repo") or "")) or "") not in blocked
        ]
    return (dl_models, blocked)


def ensure_secret(
    ns: str,
    secret_name: str,
    secret_type: str,
    prof: dict,
    cluster: str,
    krun=default_krun,
    plan_only: bool = False,
    interactive: bool = False,
    exists: bool | None = None,
    environ: dict | None = None,
    probe: bool = True,
    cred_files: dict | None = None,
) -> tuple[str, str]:
    """G10 — idempotent secret ensure that also works HEADLESS. Returns (status, message) where status ∈
    {present, created, planned, missing-source, failed, skipped}. Value resolution: well-known env/cred-file/
    profile/file (resolve_secret_value); when absent and a tty is attached, fall back to an interactive
    getpass; when absent headless, return `missing-source` with the EXACT key to set (never a mid-deploy
    surprise). `probe=False` (OFFLINE dry-run) never touches the cluster. `cred_files` overrides cred-file
    paths (tests)."""
    if not secret_name:
        return (
            "skipped",
            f"no {secret_type} secret name in profile — nothing to ensure",
        )
    if exists is None and probe:
        rc, _, _ = krun(["-n", ns, "get", "secret", secret_name])
        exists = rc == 0
    if exists:
        return ("present", f"secret '{secret_name}' already exists")
    value, source = resolve_secret_value(secret_type, prof, environ=environ, cred_files=cred_files)
    if not value and interactive and not plan_only:
        import getpass

        label = _SECRET_KINDS.get(secret_type, {}).get("label", secret_type)
        try:
            value = getpass.getpass(f"    ? {label} for secret '{secret_name}' (Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            value = ""
        if value:
            source = "interactive prompt"
    if not value:
        if plan_only and not probe:
            # OFFLINE dry-run cannot know whether the cluster Secret exists. Do not claim it is absent or emit
            # a failure-looking warning; state the conditional creation requirement instead.
            return (
                "planned",
                f"would keep secret '{secret_name}' if present; if absent, creation needs a local value — "
                f"{secret_source_hint(secret_type, cluster)}",
            )
        return (
            "missing-source",
            f"secret '{secret_name}' absent and no value source found — {secret_source_hint(secret_type, cluster)}",
        )
    if plan_only:
        return ("planned", f"would create secret '{secret_name}' from {source}")
    rc, err = _create_secret(ns, secret_name, secret_type, value, krun=krun)
    if rc == 0:
        return ("created", f"created secret '{secret_name}' from {source}")
    if "AlreadyExists" in (err or ""):
        return ("present", f"secret '{secret_name}' already exists (create race)")
    return (
        "failed",
        f"could not create secret '{secret_name}': {(err or '').strip()[:140]}",
    )


_ENSURE_GLYPH = {
    "present": "✓",
    "created": "✓",
    "planned": "•",
    "skipped": "·",
    "missing-source": "⚠",
    "failed": "✗",
    "undersized": "✗",
    "fell-back": "⚠",
    "attach-failed": "✗",
    "access-unknown": "✗",
    "access-mismatch": "✗",
}


def ensure_prerequisites(
    prof: dict,
    state: dict,
    cluster: str,
    krun=default_krun,
    applier=default_apply,
    plan_only: bool = False,
    interactive: bool = False,
    probe: bool = True,
    cred_files: dict | None = None,
) -> list[tuple[str, str]]:
    """G9+G10 — the single "ensure prerequisites" step, in dependency order: namespace (idempotent create)
    → image-pull + HF secrets (from env/cred-file/profile), BEFORE Phase C download / Phase D stage. Runs on
    EVERY path incl. the headless copy/paste flow. Prints one line per resource and returns the
    (status, message) list (for tests). Applies nothing under plan_only (dry-run / live-plan) — it prints the
    plan instead. `probe=False` (OFFLINE dry-run) additionally never touches the cluster.

    NOTE: model-cache PVC provisioning moved OUT of here to the per-recipe path (ensure_recipe_cache_pvcs,
    run AFTER cell selection in the main install flow) — one correct claim per recipe rather than a single
    profile PVC.. `applier` is retained (defaulted) for signature stability.
    """
    ns = prof.get("NAMESPACE", "")
    results: list[tuple[str, str]] = []
    print("\n── Prerequisites (namespace · secrets) ─────────────────────────────")

    ns_status, ns_msg = ensure_namespace(ns, krun=krun, plan_only=plan_only, exists=state.get("ns_ok"), probe=probe)
    print(f"  {_ENSURE_GLYPH.get(ns_status, '?')} {ns_msg}")
    results.append((ns_status, ns_msg))
    # If the namespace couldn't be ensured, the in-namespace secret ensures would cascade into failures —
    # note it once and skip them (the later preflight names the namespace as the single root cause).
    if ns_status == "failed":
        print("    (skipping secret ensure — namespace missing; the fix above is the single blocker)")
        return results

    for secret_key, secret_type in (
        ("IMAGE_PULL_SECRET", "ngc-key"),
        ("HF_SECRET", "hf-token"),
    ):
        name = prof.get(secret_key, "")
        # Reuse the Phase-A probe result when we have it (avoids a duplicate `get secret`).
        probed = {
            "IMAGE_PULL_SECRET": state.get("pull_secret_ok"),
            "HF_SECRET": state.get("hf_secret_ok"),
        }.get(secret_key)
        st, msg = ensure_secret(
            ns,
            name,
            secret_type,
            prof,
            cluster,
            krun=krun,
            plan_only=plan_only,
            interactive=interactive,
            exists=probed,
            probe=probe,
            cred_files=cred_files,
        )
        print(f"  {_ENSURE_GLYPH.get(st, '?')} {msg}")
        results.append((st, msg))
    return results


# ---------------------------------------------------------------------------
# Phase B — recipe / cell multi-select
# ---------------------------------------------------------------------------


def _cell_label(cell: dict) -> str:
    """One-line 'scenario · distribution · engine-serving' descriptor for a cell row. Pure."""
    bits = [
        cell.get("scenario", "?"),
        cell.get("distribution", "") or cell.get("goal", ""),
    ]
    eng = f"{cell.get('engine', '?')}-{cell.get('serving_mode', '?')}"
    return "  ".join([b for b in bits if b] + [eng])


# --- Grouped recipe-menu helpers (scenario → goal → model hierarchy) --------------------------------
#
# The flat numbered list was flagged as confusing in live QA. We regroup the SAME GPU-matching cells into a
# scannable scenario→goal→model tree WITHOUT changing which cells are offered. Selection numbers are assigned
# AFTER grouping (in display order) and stay stable; group-level tokens (g1 / g1a / a scenario name) are
# additive shorthand that expand to the non-installed cells nested under a header. Display + selection-index
# only — zero recipe/hash/install-logic impact.

# Friendly, operator-facing goal labels. Falls back to the raw envelope.goal (or the exemplar metric).
_GOAL_LABELS = {
    "pareto": "pareto",
    "max-concurrency-sla": "max-concurrency-sla",
    "agentic-behavior": "net-behavior",
    "throughput": "throughput",
}


def _goal_label(cell: dict) -> str:
    """Friendly goal label for a group header — from the cell's `goal` (envelope.goal) mapped to a short
    operator name, falling back to the raw goal, then the exemplar metric, then '?'. Pure.
    """
    g = (cell.get("goal") or "").strip()
    if g:
        return _GOAL_LABELS.get(g, g)
    metric = ((cell.get("exemplar") or {}).get("metric") or "").strip()
    return metric or "?"


def _replay_tag(cell: dict) -> str:
    """The dataset/trace a cell replays — its full `distribution` string (lossless w.r.t. the flat list).
    Empty distribution → '' (no tag). Pure."""
    return (cell.get("distribution") or "").strip()


def _cell_sort_key(cell: dict):
    """Within-model cell order: the base cell (no explicit -cN concurrency) first, then ascending
    concurrency, then id — so a family reads base · c16 · c32 · c64. Pure."""
    cid = cell_id(cell)
    m = re.search(r"-c(\d+)(?:-|$)", cid)
    conc = int(m.group(1)) if m else -1
    return (conc, cid)


def _short_cell_id(cell: dict) -> str:
    """Cell id with its leading '<model>-' prefix elided to '…-' (the row already sits under a model header,
    and every row shares the cluster's single GPU). Falls back to the full id when model isn't a prefix. Pure.
    """
    cid = cell_id(cell)
    model = (cell.get("model") or "").strip()
    if model and cid.startswith(model + "-"):
        return "…-" + cid[len(model) + 1 :]
    return cid


def build_recipe_menu(
    cells: list[dict],
    stamps: dict[str, dict],
) -> tuple[list[dict], dict[int, dict], dict[str, list[dict]]]:
    """PURE. Reorganize the flat GPU-matching cell list into a scannable scenario→goal→model hierarchy and
    assign each NON-installed cell a STABLE 1-based selection number (in display order). Returns
    (groups, idx_map, group_index):

      groups      ordered list of goal-level group dicts, each:
                    {token:'g1', scenario, goal, goal_label, distribution|None, models:[
                       {token:'g1a', model, cells:[ {cell, num|None, installed, status} ]} ]}
                  `distribution` is the shared replay tag when every cell in the group shares one (shown once
                  in the header), else None (each cell then carries its own tag → nothing is dropped).
      idx_map     {selection_number: cell}  — number → cell, NON-installed cells only (the parse target).
      group_index {token: [cell,...]}       — group-select shorthand → the NON-installed cells it expands to.
                  Keys (all lowercased): each goal token (g1,g2…), each model token (g1a,g1b…) and each
                  scenario name. Installed cells are excluded, so a bulk token never tries to re-install one.

    Deterministic sort: groups by (scenario, goal_label, goal), models by name, cells by _cell_sort_key.
    """
    ordered = sorted(
        cells,
        key=lambda c: (
            c.get("scenario", ""),
            _goal_label(c),
            c.get("goal", ""),
            c.get("model", ""),
            _cell_sort_key(c),
        ),
    )
    groups: list[dict] = []
    group_by_key: dict = {}
    model_by_key: dict = {}
    idx_map: dict[int, dict] = {}
    num = 0
    goal_n = 0
    for c in ordered:
        installed = cell_installed(c, stamps)
        scen = c.get("scenario", "?")
        goal = c.get("goal", "")
        gkey = (scen, goal)
        g = group_by_key.get(gkey)
        if g is None:
            goal_n += 1
            g = {
                "token": f"g{goal_n}",
                "scenario": scen,
                "goal": goal,
                "goal_label": _goal_label(c),
                "distribution": None,
                "models": [],
                "_letters": 0,
                "_dists": set(),
            }
            group_by_key[gkey] = g
            groups.append(g)
        g["_dists"].add(_replay_tag(c))
        model = c.get("model", "?")
        mkey = (gkey, model)
        m = model_by_key.get(mkey)
        if m is None:
            letter = chr(ord("a") + g["_letters"]) if g["_letters"] < 26 else f"z{g['_letters']}"
            g["_letters"] += 1
            m = {"token": g["token"] + letter, "model": model, "cells": []}
            model_by_key[mkey] = m
            g["models"].append(m)
        if installed:
            row = {
                "cell": c,
                "num": None,
                "installed": True,
                "status": c.get("status", "?"),
            }
        else:
            num += 1
            idx_map[num] = c
            row = {
                "cell": c,
                "num": num,
                "installed": False,
                "status": c.get("status", "?"),
            }
        m["cells"].append(row)

    group_index: dict[str, list[dict]] = {}
    for g in groups:
        dists = g.pop("_dists")
        g.pop("_letters")
        g["distribution"] = next(iter(dists)) if len(dists) == 1 else None
        g_cells = [r["cell"] for m in g["models"] for r in m["cells"] if not r["installed"]]
        group_index[g["token"].lower()] = g_cells
        group_index.setdefault(g["scenario"].lower(), []).extend(g_cells)
        for m in g["models"]:
            group_index[m["token"].lower()] = [r["cell"] for r in m["cells"] if not r["installed"]]
    return groups, idx_map, group_index


def parse_recipe_selection(
    raw: str,
    idx_map: dict[int, dict],
    group_index: dict[str, list[dict]],
) -> tuple[list[dict], str, list[tuple[str, list[str]]]]:
    """PURE. Resolve a raw selection string to (cells, error, expansions). Accepts, comma-separated:
    numbers (5), ranges (5-8), and group tokens / scenario names (g1, g1a, llm-perf → every
    NON-installed cell nested under that header); plus the whole-list words 'all'/'' and the skip words
    'none'/'skip'/'0'. De-dups (first-seen order). idx_map/group_index only hold non-installed cells, so an
    installed (🔒) cell is never selectable and a bulk token silently skips it. Returns error='' on success,
    else a message + []. `expansions` lists (token, [cell_id,…]) for each GROUP/scenario token that matched,
    so the caller can echo what a bulk token expanded to."""
    raw = (raw or "").strip()
    low = raw.lower()
    if low in ("", "all"):
        return (list(idx_map.values()), "", [])
    if low in ("none", "skip", "0"):
        return ([], "", [])
    out: list[dict] = []
    seen: set = set()
    expansions: list[tuple[str, list[str]]] = []

    def _add(c: dict) -> None:
        k = c.get("_path", "") or cell_id(c)
        if k not in seen:
            seen.add(k)
            out.append(c)

    for tok in [t.strip() for t in raw.split(",") if t.strip()]:
        mrange = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tok)
        if mrange:
            lo, hi = int(mrange.group(1)), int(mrange.group(2))
            if lo > hi:
                lo, hi = hi, lo
            bad = [n for n in range(lo, hi + 1) if n not in idx_map]
            if bad:
                return ([], f"range {tok!r}: no recipe number {bad}", [])
            for n in range(lo, hi + 1):
                _add(idx_map[n])
            continue
        if tok.isdigit():
            n = int(tok)
            if n not in idx_map:
                return ([], f"no recipe number {n}", [])
            _add(idx_map[n])
            continue
        tl = tok.lower()
        if tl in group_index:
            grp = group_index[tl]
            for c in grp:
                _add(c)
            expansions.append((tok, [cell_id(c) for c in grp]))
            continue
        return ([], f"unknown selection {tok!r}", [])
    return (out, "", expansions)


def resolve_recipe_selection(
    cells: list[dict],
    stamps: dict[str, dict],
    recipes_arg: str | None,
    all_matching: bool,
    reinstall: bool = False,
) -> tuple[list[dict], list[str]]:
    """Headless cell selection. Returns (selected_cells, errors). Pure.

      --all-matching     → every GPU-matching cell not already installed.
      --recipes a,b,c    → cells matched by catalog name, catalog-relative/absolute path, or _path basename;
                           unknown names → error.

    IDEMPOTENCY: an already-installed cell is DROPPED (a no-op), for `--recipes` exactly as for
    `--all-matching`. It used to be dropped only by `--all-matching`, so `--recipes <already-installed>` —
    the obvious thing to type when re-running the documented command — re-staged and re-offered the download.
    Being named explicitly is not consent to redo the work; `--reinstall` is. Dropped cells are returned to
    the caller via `errors`-free channel: the caller prints them, so this is never a silent skip.
    """
    errors: list[str] = []
    if all_matching:
        return [c for c in cells if not cell_installed(c, stamps)], errors

    if recipes_arg:
        want = [t.strip() for t in recipes_arg.split(",") if t.strip()]
        # Exact catalog-name/path indexes (unique) + a basename-alias index that keeps ALL cells sharing a
        # basename. Two cells can have distinct catalog names but the SAME dir basename (e.g. two scenarios
        # with different catalog names may both live at .../nemotron-ultra-3-b200-vllm-agg-pareto).
        # The old single dict silently overwrote on basename collision → --recipes <basename> resolved to
        # whichever cell was iterated last (wrong cell, no warning). Now: exact name wins; an ambiguous
        # basename is a hard error that names the candidates.
        by_name: dict[str, dict] = {}
        by_path: dict[str, dict] = {}
        by_basename: dict[str, list[dict]] = {}
        for c in cells:
            by_name[cell_id(c)] = c
            rel = Path(c.get("_path", ""))
            # Accept exactly what every other lifecycle verb accepts: the catalog-relative cell path or its
            # absolute form. `strict=False` avoids requiring a synthetic test fixture to exist on disk.
            by_path[str(rel).rstrip("/")] = c
            by_path[str((ROOT / rel).resolve(strict=False)).rstrip("/")] = c
            by_basename.setdefault(rel.name, []).append(c)
        selected: list[dict] = []
        skipped: list[str] = []
        seen: set[str] = set()
        for tok in want:
            clean_tok = tok.rstrip("/")
            c = by_name.get(tok) or by_path.get(clean_tok)
            if c is None and Path(clean_tok).is_absolute():
                c = by_path.get(str(Path(clean_tok).resolve(strict=False)).rstrip("/"))
            if c is None:
                # Not an exact catalog name — fall back to the basename alias, but only if unambiguous.
                matches = by_basename.get(tok, [])
                if len(matches) == 1:
                    c = matches[0]
                elif len(matches) > 1:
                    names = ", ".join(sorted(cell_id(m) for m in matches))
                    errors.append(
                        f"ambiguous recipe '{tok}' — {len(matches)} cells share that directory basename: "
                        f"{names}. Use the full cell name to disambiguate."
                    )
                    continue
                else:
                    errors.append(
                        f"no GPU-matching cell named or located at '{tok}' "
                        "(use its catalog name, cell directory basename, or full recipe path)"
                    )
                    continue
            key = c.get("_path", "")
            if key in seen:
                continue
            seen.add(key)
            if cell_installed(c, stamps) and not reinstall:
                skipped.append(cell_id(c))
                continue
            selected.append(c)
        for name in skipped:
            print(
                f"  {GLYPH_INSTALLED} {name}: already installed on this cluster — nothing to do "
                f"(re-run with --reinstall to redo staging + preflight)"
            )
        return selected, errors

    return [], errors


def list_recipes(cells: list[dict], stamps: dict[str, dict], cluster: str, gpu: str) -> None:
    """--list-recipes: print the GPU-compat matrix + which cells are already installed on this cluster, then
    the caller exits. Read-only; no cluster probing."""
    print(f"\n── Recipes matching {gpu or '?'} on {cluster} " + "─" * max(4, 40 - len(gpu or "?") - len(cluster)))
    if not cells:
        print(f"  (no catalog cells target {gpu or 'this GPU'} — nothing to install here)")
        return
    # No `status` column. envelope.status is an INTERNAL maturity ladder (planned -> wip -> runs ->
    # performant -> exemplar) that gates provenance-check / pooled-check. Surfaced HERE, next to a column
    # literally headed "install", an operator reads "[wip]" as "this recipe is not ready to run" -- which is
    # not what it means. It means "no published numbers yet". The only thing that belongs in an install
    # picker is whether the recipe is installed.
    print(f"  {'':<2} {'cell':<52}  install")
    print(f"  {'':<2} {'─'*52}  {'─'*22}")
    for _i, c in enumerate(cells, 1):
        installed = cell_installed(c, stamps)
        mark = GLYPH_INSTALLED if installed else " "
        inst = "installed" if installed else "—"
        cid = cell_id(c)
        cid_disp = cid if len(cid) <= 52 else cid[:49] + "..."
        print(f"  {mark:<2} {cid_disp:<52}  {inst}")
    n_inst = sum(1 for c in cells if cell_installed(c, stamps))
    print(f"\n  {len(cells)} matching · {n_inst} installed · {len(cells) - n_inst} available")
    print(f"  Install:  llmb-k8s install {cluster} --recipes <cell>[,<cell>...]   (or --all-matching)")


# ---------------------------------------------------------------------------
# Cluster-truth INSTALLED state (shared with `fleet`)
# ---------------------------------------------------------------------------
#
# "Is this already installed?" must be answerable from the CLUSTER, not from a local stamp file: a model
# downloaded from another worktree, by a colleague, or before a fresh clone is still installed. fleet already
# answers exactly that (fleet_render.discover_model_caches / discover_installed) with HONEST states — a
# Bound-but-unvouched-for PVC reads `contents unverified`, never ✓. We call THOSE functions rather than
# re-deriving a second, kinder answer here: two install-state definitions is how a green-signal-that-isn't
# gets shipped.
#
# The local stamp keeps its separate job (it records STAGING + preflight, which no cluster object shows) and
# still decides selectability. The cluster read is ADDITIVE VISIBILITY.

_CACHE_GLYPH = {"ready": GLYPH_READY, "warn": GLYPH_NEEDS, "failed": GLYPH_BLOCKED}


def probe_cluster_install_state(prof: dict, krun=default_krun) -> dict:
    """LIVE read of what is already installed in this profile's namespace. Returns
    {"caches": {pvc_name: {state, why}}, "cells": {cell_name: {state, why}}, "readable": bool}.

    `readable` is False when the reads were denied/unavailable — the caller must then say NOTHING about
    install state rather than implying "absent" (an RBAC denial is not an empty cluster).
    """
    ns = (prof.get("NAMESPACE") or "").strip()
    out = {"caches": {}, "cells": {}, "readable": False}
    if not ns:
        return out
    try:
        import fleet_render as _fr
    except Exception:
        return out

    def _get(kind: str):
        rc, so, _ = krun(["-n", ns, "get", kind, "-o", "json"], timeout=30)
        if rc != 0:
            return None
        try:
            return json.loads(so)
        except ValueError:
            return None

    pvcs_j, jobs_j, deploys_j = _get("pvc"), _get("jobs"), _get("deploy")
    if pvcs_j is None and jobs_j is None and deploys_j is None:
        return out
    out["readable"] = True
    jobs = (jobs_j or {}).get("items") or []
    # Keep each claim's LABELS. The cache verdict below is about the CLAIM ("these weights finished
    # downloading"), which says nothing about WHICH model finished — so the panel needs the per-model
    # stamps to avoid reporting one model's completion as every model's.
    out["cache_labels"] = {}
    for it in (pvcs_j or {}).get("items") or []:
        md = it.get("metadata") or {}
        out["cache_labels"][md.get("name") or ""] = md.get("labels") or {}
    for c in _fr.discover_model_caches(pvcs_j, jobs):
        if c.get("kind") != "cache":
            continue  # per-run `-artifacts` output + the `llmb-control` state PVC are not model caches
        # discover_model_caches names the row by MODEL when a download vouches for it, so key on the pvc named
        # in `why` as well as the row name — the caller looks up by the claim its recipe derived.
        out["caches"][c["name"]] = {"state": c["state"], "why": c["why"]}
        why = c.get("why") or ""
        if "pvc " in why:
            claim = why.split("pvc ", 1)[1].split(" ")[0].strip()
            out["caches"].setdefault(claim, {"state": c["state"], "why": c["why"]})
    for d in _fr.discover_installed(deploys_j, jobs_j):
        if d.get("kind") == "cell":
            out["cells"][d["name"]] = {"state": d["state"], "why": d["why"]}
    return out


def cell_cluster_state(cell: dict, prof: dict, live: dict, root: Path = ROOT) -> tuple[str, str]:
    """PURE. (glyph, one-line description) of what the CLUSTER says about this cell: its model cache and
    whether a server is deployed. Returns ('', '') when the cluster could not be read — we never render a
    guess as a fact. The cache verdict is fleet's, verbatim (ready / downloading / contents unverified /
    not Bound), so install and fleet can never disagree about the same PVC."""
    if not live.get("readable"):
        return ("", "")
    claim = resolve_cache_claim(cell, prof)[0]
    if not claim:
        # Without a configured claim, report configuration state rather than cache absence.
        return (
            GLYPH_NEEDS,
            "model cache: NOT CONFIGURED for this model "
            f"(set {model_cache_env_key(cell.get('model') or '')} or MODEL_CACHE_PVC)",
        )
    c = (live.get("caches") or {}).get(claim)
    # A claim-level completion stamp may describe a different model on a shared cache. Require a matching
    # per-model stamp before applying the claim verdict to this cell.
    model = cell.get("model") or ""
    labels = (live.get("cache_labels") or {}).get(claim)
    this_model_stamped = None  # None = we have no labels to judge by
    if labels is not None and model:
        slug = _mc.model_cache_slug(model)
        this_model_stamped = (
            any(_mc.model_from_stamp_key(k) == slug for k in labels)
            or _mc.model_cache_slug(labels.get("llmb.nvidia.com/model-name") or "") == slug
        )
    if c is None:
        cache = "model cache: absent"
    elif c["state"] == "ready" and this_model_stamped is False:
        cache = (
            f"model cache '{claim}': present, but nothing on it vouches for {model} "
            f"(no per-model stamp) — run install to verify"
        )
    elif c["state"] == "ready":
        cache = "model cache: downloaded"
    elif c["state"] == "failed":
        cache = f"model cache: {c['why'].split(' · ')[0]}"
    else:
        cache = "model cache: " + ("downloading" if "downloading" in c["why"] else "contents UNVERIFIED")
    srv = live["cells"].get(cell_id(cell))
    server = f" · server: {srv['why']}" if srv else ""
    ready_here = bool(c and c["state"] == "ready" and this_model_stamped is not False)
    glyph = GLYPH_READY if ready_here else (GLYPH_NEEDS if c else " ")
    return (glyph, cache + server)


def render_installed_panel(cells: list[dict], stamps: dict[str, dict], prof: dict, live: dict, cluster: str) -> str:
    """PURE. The 'what is ALREADY here' panel printed BEFORE the picker, so the user chooses with state in
    in front of them. Never claims installed for something we could not verify."""
    head = f"── Already installed on {cluster} "
    lines = ["", head + "─" * max(4, 68 - len(head))]
    if not live.get("readable"):
        lines.append(
            "  ("
            + (live.get("skip_reason") or "cluster not readable right now")
            + " — showing local install stamps only; nothing below is a claim"
        )
        lines.append("   about what is actually on the cluster)")
    else:
        # SAY WHERE THE EVIDENCE COMES FROM. Two different sources are interleaved on every row and they
        # answer different questions: a local stamp records that THIS checkout staged the cell, while the
        # cache verdict is read from the PVC's labels. Neither one opens the PVC and looks at the weights —
        # that is the on-PVC probe, which runs during model selection. An operator reading
        # "model cache: downloaded" as "the bytes are verified present" is the mistake this line prevents.
        lines.append(
            "  (evidence: 'local stamp' = this checkout staged it · cache state = the PVC's " "download stamp,"
        )
        lines.append("   not a read of the weights themselves)")
    rows = []
    for c in cells:
        stamped = cell_installed(c, stamps)
        glyph, desc = cell_cluster_state(c, prof, live)
        # Show a row whenever we know ANYTHING real about this cell here: a local stamp, or any cluster-side
        # cache/server state — INCLUDING "contents UNVERIFIED". Hiding the unverified case is what makes a
        # half-installed cluster look pristine and sends the user round the download loop again.
        if not stamped and desc in ("", "model cache: absent"):
            continue
        marks = []
        if stamped:
            marks.append("staged+preflighted (local stamp)")
        if desc:
            marks.append(desc)
        cid = cell_id(c)
        cid = cid if len(cid) <= 52 else cid[:49] + "..."
        rows.append(f"  {glyph or GLYPH_INSTALLED} {cid:<52} {' · '.join(marks)}")
    if not rows:
        lines.append("  (nothing installed here yet — every matching recipe below is a fresh install)")
    else:
        lines += rows
    return "\n".join(lines)


def phase_b_recipe_selection(
    cells: list[dict],
    stamps: dict[str, dict],
    plan_only: bool,
) -> list[dict]:
    """Interactive recipe/cell multi-select. Already-installed cells render GREYED-OUT / non-selectable.
    Reuses install.py's numbered comma-select widget (Teleport/non-tty safe — NOT questionary). Returns the
    list of selected cell dicts."""
    print("\n── Recipe selection ───────────────────────────────────────────────")
    print(f"  {GLYPH_INSTALLED} = already installed (non-selectable)  ·  cells matching this cluster's GPU")

    groups, idx_map, group_index = build_recipe_menu(cells, stamps)

    for g in groups:
        dist = f"  ({g['distribution']})" if g["distribution"] else ""
        print()
        print(f"  [{g['token']}] {g['scenario']}  ·  goal: {g['goal_label']}{dist}")
        for m in g["models"]:
            print(f"    [{m['token']}] {m['model']}")
            for row in m["cells"]:
                c = row["cell"]
                sid = _short_cell_id(c)
                eng = f"{c.get('engine', '?')}-{c.get('serving_mode', '?')}"
                # When the group shares one distribution it's in the header; else show each cell's own tag
                # so no dataset/distribution info is lost relative to the flat list.
                tag = _replay_tag(c)
                extra = eng if g["distribution"] is not None else (f"{tag}  {eng}" if tag else eng)
                if row["installed"]:
                    print(f"      {GLYPH_INSTALLED} {'--':>3}  {sid:<40} {extra}")
                else:
                    print(f"         {row['num']:>3}  {sid:<40} {extra}")
    print()

    if not idx_map:
        print("  All GPU-matching cells are already installed. Nothing to do.")
        return []

    selectable = list(idx_map.values())
    if plan_only:
        print("  [plan] would prompt for recipe selection. Selecting all installable cells.")
        return selectable

    print("Select recipes to set up:")
    print(
        "  numbers 1,3  ·  ranges 5-8  ·  group tokens g1 / g1a  ·  a scenario name  ·  "
        "'all'  ·  Enter / 'none' to skip"
    )
    while True:
        raw = input("  ? [all]: ").strip()
        chosen, err, expansions = parse_recipe_selection(raw, idx_map, group_index)
        if err:
            print(f"    {err}. Use numbers (1,3), ranges (5-8), a group token (g1/g1a), " f"a scenario name, or 'all'.")
            continue
        for tok, ids in expansions:
            skipped = "" if ids else "  (all installed — nothing to add)"
            print(f"    {tok} → selected {len(ids)} cell(s): {', '.join(ids) or '—'}{skipped}")
        return chosen


# Cluster-profile selection for an argument-free `llmb-k8s install`.
# Interactive sessions select a profile; non-interactive sessions receive a
# clear command-line remedy instead of waiting for input.
#   • ZERO profiles → there is nothing to pick; point at `llmb-k8s init`, exit non-zero.

_PICK_QUIT = "__quit__"


def current_kube_context(krun=default_krun) -> str:
    rc, out, _ = krun(["config", "current-context"], timeout=10)
    return (out or "").strip() if rc == 0 else ""


def default_profile(
    profiles: list[str],
    environ: dict | None = None,
    current_ctx: str = "",
    profiles_dir: Path | None = None,
) -> str:
    """PURE-ish (reads profile files only). The Enter-default cluster: an explicit LLMB_CLUSTER, else the
    profile that pins the CURRENT kube context (you just logged into it), else the first profile. ALWAYS
    returns a member of `profiles` (or '' when there are none) — an unselectable default is a hang.
    """
    if not profiles:
        return ""
    environ = os.environ if environ is None else environ
    want = (environ.get("LLMB_CLUSTER") or "").strip()
    if want in profiles:
        return want
    if current_ctx:
        d = profiles_dir or _pr.PROFILES_DIR
        for p in profiles:
            if (_pr.profile_context(p, d) or "") == current_ctx:
                return p
    return profiles[0]


def render_profile_menu(profiles: list[str], default: str, profiles_dir: Path | None = None) -> str:
    """PURE. Numbered cluster-profile menu; the Enter-default marked. Each row shows the profile's GPU +
    namespace so the choice is recognizable without opening the .env."""
    d = profiles_dir or _pr.PROFILES_DIR
    if not profiles:
        return "  (no cluster profiles yet)"
    lines = ["  Cluster profiles (cluster-profiles/<name>.env):"]
    w = len(str(len(profiles)))
    for i, p in enumerate(profiles, 1):
        env = _pr._read_env(_pr.profile_env_path(p, d))
        gpu = _pr.profile_gpu_type(env) or "?"
        ns = env.get("NAMESPACE", "") or "?"
        mark = "  (press Enter)" if p == default else ""
        lines.append(f"    {i:>{w}}  {p:<28} {gpu:<8} ns={ns}{mark}")
    return "\n".join(lines)


def pick_profile_interactive(profiles: list[str], default: str, prompt=None, profiles_dir: Path | None = None) -> str:
    """Numbered profile picker. Enter takes `default`; a number picks; 'q' quits. Terminates on EOF (a piped
    stdin that runs dry must not spin forever) by taking the default.

    `prompt` is resolved LATE (None → the current builtins.input) so a test — or anything that monkeypatches
    input — is actually honored; a `prompt=input` default would bind the original at def time and silently
    ignore the patch."""
    prompt = prompt or input
    print("\n── Cluster ────────────────────────────────────────────────────────")
    print(render_profile_menu(profiles, default, profiles_dir))
    while True:
        try:
            raw = prompt(f"\n? Which cluster? [1-{len(profiles)}]  (q = quit) [{default}]: ").strip()
        except EOFError:
            return default
        if not raw:
            return default
        if raw.lower() in ("q", "quit"):
            return _PICK_QUIT
        if raw.isdigit() and 1 <= int(raw) <= len(profiles):
            return profiles[int(raw) - 1]
        if raw in profiles:
            return raw
        print(f"    Enter 1–{len(profiles)}, a profile name, or q. (Enter alone = {default})")


def resolve_cluster_arg(
    cluster: str | None,
    *,
    is_tty: bool,
    profiles: list[str],
    krun=default_krun,
    prompt=None,
    profiles_dir: Path | None = None,
) -> tuple[str, str]:
    """Decide which cluster to install onto. Returns (cluster, error_message); error_message non-empty means
    stop. Never prompts when `cluster` was given (the existing contract) or when stdin is not a tty.
    """
    if cluster:
        return cluster, ""
    if not profiles:
        return "", (
            "install: no cluster profiles yet — there is nothing to install onto.\n"
            "  Create one first (discovers your connected clusters, no arguments needed):\n"
            "    scripts/llmb-k8s init"
        )
    if len(profiles) == 1:
        print(f"  Using the only cluster profile: {profiles[0]}")
        return profiles[0], ""
    default = default_profile(profiles, current_ctx=current_kube_context(krun), profiles_dir=profiles_dir)
    if not is_tty:
        return "", (
            "install: no cluster given and stdin is not a terminal, so I will not prompt.\n"
            f"  Pass one:  scripts/llmb-k8s install <cluster-profile>   "
            f"(or --cluster <name>)\n"
            f"  Available: {', '.join(profiles)}"
        )
    if (os.environ.get("LLMB_CLUSTER") or "").strip() in profiles:
        print(f"  Using cluster profile from $LLMB_CLUSTER: {default}")
        return default, ""
    picked = pick_profile_interactive(profiles, default, prompt=prompt, profiles_dir=profiles_dir)
    if picked == _PICK_QUIT:
        return "", "  Cancelled."
    return picked, ""


# ---------------------------------------------------------------------------
# Phase A — profile review panel
# ---------------------------------------------------------------------------


def phase_a_review(
    cluster: str,
    prof: dict,
    resolution: _pr.Resolution,
    state: dict,
    dry_run: bool,
) -> bool:
    """Print Phase A panel. Return True if user confirms to proceed."""
    ns = prof.get("NAMESPACE", "")
    ctx = prof.get("KUBE_CONTEXT", "")
    gpu_product = prof.get("GPU_PRODUCT", "")
    pull_secret = prof.get("IMAGE_PULL_SECRET", "")
    hf_secret = prof.get("HF_SECRET", "")
    pvc = prof.get("MODEL_CACHE_PVC", "")

    bar = "─" * max(4, 66 - len(f"Profile: {cluster}"))
    print(f"\n── Profile: {cluster} {bar}")

    # Context
    if ctx:
        if dry_run:
            ctx_line = f"{ctx}   (not probed — offline plan)"
        else:
            ctx_line = f"{ctx}   ✓ reachable" if resolution.ok else f"{ctx}   ✗ unreachable"
    else:
        ctx_line = "(none pinned — using current kubectl context)"
    print(f"  Context      {ctx_line}")

    # Namespace
    ns_ok = state.get("ns_ok")
    if ns_ok is True:
        ns_line = f"{ns or '?'}   ✓ exists"
    elif ns_ok is False:
        ns_line = f"{ns or '?'}   ✗ not found"
    else:
        ns_line = f"{ns or '?'}   (unknown)"
    print(f"  Namespace    {ns_line}")

    # GPU nodes
    total = state.get("gpu_nodes_total")
    used = state.get("gpu_nodes_used")
    if total is not None:
        gpu_line = f"{gpu_product}   ✓ {total} nodes / {used or 0} in use"
    else:
        gpu_line = f"{gpu_product or '?'}   (unknown)"
    print(f"  GPU          {gpu_line}")

    # Pull secret
    ps_ok = state.get("pull_secret_ok")
    if not pull_secret:
        ps_line = "(not configured)"
    elif ps_ok is True:
        ps_line = f"{pull_secret}   ✓ valid"
    elif ps_ok is False:
        ps_line = f"{pull_secret}   – not found"
    else:
        ps_line = f"{pull_secret}   (unknown)"
    print(f"  Pull secret  {ps_line}")

    # HF secret
    hf_ok = state.get("hf_secret_ok")
    if not hf_secret:
        hf_line = "(not configured)"
    elif hf_ok is True:
        hf_line = f"{hf_secret}   ✓ valid"
    elif hf_ok is False:
        hf_line = f"{hf_secret}   – not found"
    else:
        hf_line = f"{hf_secret}   (unknown)"
    print(f"  HF secret    {hf_line}")

    # PVC
    pvc_phase = state.get("pvc_phase")
    pvc_free = state.get("pvc_free_gib")
    if not pvc:
        pvc_line = "(not configured)"
    elif pvc_phase == "Bound":
        free_str = f"  /  {pvc_free / 1024:.1f} TiB free" if pvc_free is not None else ""  # pvc_free is GiB (audit #4)
        pvc_line = f"{pvc}   ✓ Bound{free_str}"
    elif pvc_phase:
        pvc_line = f"{pvc}   ✗ {pvc_phase}"
    else:
        pvc_line = f"{pvc}   (unknown)"
    print(f"  Model PVC    {pvc_line}")

    print()
    print("  To change any of these values:")
    print(f"    $EDITOR cluster-profiles/{cluster}.env")
    print(f"    then re-run: llmb-k8s install --cluster {cluster}")
    print()

    if dry_run:
        print("  (auto-confirmed — skipping the profile-review prompt)")
        return True

    ans = input("? Proceed with this profile? [Y/n]: ").strip().lower()
    return ans in ("", "y", "yes")


# ---------------------------------------------------------------------------
# Phase B — model selection
# ---------------------------------------------------------------------------


def phase_b_model_selection(
    models: list[dict],
    free_gib: float | None,
    pvc_name: str,
    on_pvc_status: dict[str, str],
    dry_run: bool,
) -> list[dict]:
    """Interactive model selection. Returns list of selected model dicts."""
    pvc_label = f"  PVC: {pvc_name}"
    if free_gib is not None:
        total_str = f"{free_gib:.1f} GiB" if free_gib < 1000 else f"{free_gib / 1024:.1f} TiB"
        pvc_label += f"  /  {total_str} free"

    print("\n── Model selection ────────────────────────────────────────────────")
    print(pvc_label)
    print()

    # Refuse models without a pinned revision.
    valid_models = []
    for m in models:
        if not m.get("model_revision"):
            print(
                f"  ⚠  {m['model_name']}: no pinned model_revision — skipping "
                f"(pin serving.model_revision in recipe.yaml first)"
            )
        else:
            valid_models.append(m)

    if not valid_models:
        print("  No models with pinned revisions found in catalog.")
        return []

    # Header
    print(f"  {'Model':<24} {'Size':>8}  {'Recipes':>7}  On PVC")
    print(f"  {'─'*24} {'─'*8}  {'─'*7}  {'─'*20}")

    selectable: list[tuple[int, dict]] = []
    already_installed: list[dict] = []

    for m in valid_models:
        sz = model_size_gib(m)
        sz_str = f"~{sz} GB" if sz else "?"
        status = on_pvc_status.get(_norm_repo(m["model_repo"]), "unknown")
        recipes = m.get("recipe_count", 0)

        # States come from model_cache.cache_completeness. 'installed'/'not found' are accepted as aliases
        # so an older caller (or a test) still renders correctly.
        if status in (_mc.STATE_COMPLETE, "installed"):
            print(f"  {'✓ complete':<12}  {m['model_name']:<20} {sz_str:>8}  {recipes:>7}")
            already_installed.append(m)
        elif status == _mc.STATE_PRESENT_UNVERIFIED:
            # Keep present-but-unverified distinct from complete and incomplete. Offer explicit verification
            # or re-download rather than assuming either state.
            idx = len(selectable) + 1
            selectable.append((idx, m))
            print(
                f"  ? {idx}: {m['model_name']:<20} {sz_str:>8}  {recipes:>7}  "
                f"present, completeness UNPROVEN — select to re-download, or leave it and run "
                f"`llmb-k8s preflight` to verify in place"
            )
        elif status == _mc.STATE_INCOMPLETE:
            idx = len(selectable) + 1
            selectable.append((idx, m))
            print(f"  ⚠ {idx}: {m['model_name']:<20} {sz_str:>8}  {recipes:>7}  " f"⚠ incomplete (re-select to resume)")
        else:
            idx = len(selectable) + 1
            selectable.append((idx, m))
            not_found = "– absent" if status in (_mc.STATE_ABSENT, "not found") else f"({status})"
            print(f"    {idx}: {m['model_name']:<20} {sz_str:>8}  {recipes:>7}  {not_found}")

    # LEGEND. "(unknown)" and "?" are the honest answers to two different questions, and an operator
    # reading a column of them cannot tell whether the tool is broken, stalled, or telling the truth.
    # Say which it is, and say that neither one silently skips a download.
    _unknowns = [
        m for m in valid_models if on_pvc_status.get(_norm_repo(m["model_repo"]), "unknown") == _mc.STATE_UNKNOWN
    ]
    _nosize = [m for m in valid_models if not model_size_gib(m)]
    if _unknowns:
        print(f"  (unknown) = the on-PVC probe could not answer for {len(_unknowns)} model(s) — the PVC was")
        print("             unreachable or the check pod did not schedule. Treated as NOT installed, so a")
        print("             download is offered rather than silently skipped. Re-run to re-probe.")
    if _nosize:
        print(f"  ?         = size not in the known-size table for {len(_nosize)} model(s); the download is")
        print("             still exact (it pulls the pinned revision), only this estimate is missing.")
    if _unknowns or _nosize:
        print()

    print()

    if not selectable:
        print("  All models already installed. Nothing to do.")
        return []

    if dry_run:
        print("  [plan] would prompt for model selection. Selecting all installable models.")
        return [m for _, m in selectable]

    # Prompt for selection.
    print("Select models to install (comma-separated numbers, or 'all', or empty to skip):")
    idx_map = dict(selectable)
    available_idxs = [str(idx) for idx, _ in selectable]

    while True:
        raw = input("  ? [all]: ").strip()
        if raw == "" or raw.lower() == "all":
            selected = [m for _, m in selectable]
            break
        if raw.lower() in ("none", "skip", "0"):
            selected = []
            break
        try:
            chosen = [int(x.strip()) for x in raw.split(",")]
            bad = [c for c in chosen if c not in idx_map]
            if bad:
                print(f"    Invalid selection(s): {bad}. Choose from: {available_idxs}")
                continue
            selected = [idx_map[c] for c in chosen]
            break
        except ValueError:
            print("    Enter numbers separated by commas (e.g. 1,2), 'all', or press Enter.")

    if not selected:
        print("  No models selected.")
        return []

    # Space check
    sizes = [model_size_gib(m) for m in selected]
    projected, will_fit = pvc_space_math(free_gib, sizes)

    print()
    print("  Selected:")
    running_free = free_gib
    for m, sz in zip(selected, sizes, strict=True):
        sz_str = f"~{sz} GB" if sz else "?"
        if running_free is not None and sz is not None:
            running_free -= sz
            fit_str = f"→ {running_free / 1024:.1f} TiB remaining" if running_free >= 0 else "✗ insufficient space"
        else:
            fit_str = ""
        print(f"    [x] {m['model_name']:<24} {sz_str:>8}  {fit_str}")

    if not will_fit:
        total_needed = sum(s for s in sizes if s is not None)
        verdict = pvc_space_verdict(projected, total_needed)
        deficit = -projected if projected is not None else 0
        print()
        print("  ✗ Insufficient PVC space for selected models.")
        print(f"    Projected free after install: {projected:.1f} GiB  (short by ~{deficit:.0f} GiB)")
        if verdict == "block":
            # HARD BLOCK — a hundreds-of-GiB deficit is a guaranteed ENOSPC. Do NOT offer a proceed prompt
            # that would launch a doomed multi-minute download. Refuse with remediation.
            print("    This download CANNOT fit and would fail with ENOSPC after a long partial download.")
            print(
                f"    Remediation: free ≥{deficit:.0f} GiB on the cache PVC, resize/expand it, or select "
                f"fewer models. Not proceeding."
            )
            return []
        # Marginal overage only (≤ a few %): the model likely still fits within reported headroom — warn+proceed.
        print("    (marginal overage — reported free may understate real headroom)")
        ans = input("  Proceed anyway? [y/N]: ").strip().lower()
        if ans not in ("y", "yes"):
            return []

    return selected


# ---------------------------------------------------------------------------
# Phase C — download
# ---------------------------------------------------------------------------

# The Job name is `llmb-download-<slug>-<rev12>`; k8s caps resource names at 63 chars. Budget the model
# slug so the total fits: 63 - len("llmb-download-") - len("-<12-char rev>") = 63 - 14 - 13 = 36. This helper
# is the SINGLE SOURCE for the slug — both the template's metadata.name and the job_name install.py builds to
# query that Job go through it, so they can never drift (audit #11: name was left unsanitized after the label).
_NAME_SLUG_BUDGET = 63 - len("llmb-download-") - 13


def _model_job_slug(model_name: str) -> str:
    """RFC-1123-safe, length-budgeted slug for a model's download Job name. Pure."""
    s = re.sub(r"[^a-z0-9-]+", "-", (model_name or "").lower()).strip("-")
    return s[:_NAME_SLUG_BUDGET].strip("-") or "model"


def _kubectl_hint(prof: dict, ns: str) -> str:
    """The `kubectl` prefix an OPERATOR must type to reach the SAME cluster install just talked to.

    install pins --context from the profile on every call it makes itself, but the commands it PRINTS
    were bare `kubectl -n <ns> ...`. On a workstation with several kubeconfig contexts that silently
    resolves to whatever context is current -- a different cluster -- and the operator gets

        error: error from server (NotFound): jobs.batch "llmb-download-..." not found in namespace "..."

    for a Job that exists and is running. The namespace in the error is right, which makes it read like
    the tool lied about the Job name. Pin the context in the hint so a copy-paste lands where install did.
    """
    ctx = (prof.get("KUBE_CONTEXT") or "").strip()
    return f"kubectl --context {ctx} -n {ns}" if ctx else f"kubectl -n {ns}"


def render_download_job(
    model: dict,
    prof: dict,
    tmpl_path: Path = DOWNLOAD_TMPL,
) -> str:
    """Render the Jinja2 download Job manifest for a model."""
    env = Environment(
        loader=FileSystemLoader(str(tmpl_path.parent)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template(tmpl_path.name)
    subpath = (prof.get("MODEL_CACHE_SUBPATH") or ".").strip() or "."
    return tmpl.render(
        model_repo=model["model_repo"],
        model_revision=model["model_revision"],
        model_name=model["model_name"],
        job_name_slug=_model_job_slug(model["model_name"]),  # audit #11: RFC-1123-safe, length-budgeted
        model_cache_subpath=subpath,
        # Size the deadline and memory from the model estimate; unknown sizes use conservative floors.
        active_deadline_s=download_deadline_s(model_size_gib(model)),
        download_mem_request=f"{download_mem_request_gib(model_size_gib(model))}Gi",
        download_mem_limit=f"{download_mem_limit_gib(model_size_gib(model))}Gi",
        # Pass profile vars so the template can reference them (optional convenience).
        namespace=prof.get("NAMESPACE", ""),
        image_pull_secret=prof.get("IMAGE_PULL_SECRET", ""),
        hf_secret=prof.get("HF_SECRET", "hf-token-secret"),
        model_cache_pvc=prof.get("MODEL_CACHE_PVC", ""),
        # Placement for the claim this Job mounts. Rendered as an indented YAML block (or "" when the
        # profile sets no selector, so clusters whose cache mounts everywhere render byte-identically).
        model_cache_node_selector="\n".join(
            f'        {k}: "{v}"' for k, v in sorted(_mc.cache_pod_placement(prof)[0].items())
        ),
    )


def _envsubst_profile(manifest: str, prof: dict) -> str:
    """Replace ${VAR} placeholders in manifest with profile values."""
    import re

    def _sub(m):
        key = m.group(1)
        return prof.get(key, m.group(0))

    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", _sub, manifest)


def _pvc_parallel_safe(ns: str, pvcs: set[str], krun=default_krun) -> tuple[bool, str]:
    """Can several download Jobs run AT THE SAME TIME against these claims? → (ok, why).

    Downloads used to run strictly one-at-a-time: apply, block ~45+ min, apply the next. For three models
    that delays independent work. They are separate Jobs writing to
    separate subpaths, so k8s will happily run them concurrently -- with one hard constraint:

      * distinct PVCs           -> always safe (independent volumes).
      * one shared RWX claim    -> safe (ReadWriteMany is exactly this case).
      * one shared RWO claim    -> NOT safe. ReadWriteOnce binds to a single node; the second pod either
                                  co-schedules by luck or sits Pending forever. Sequential is CORRECT there,
                                  and silently parallelising would look like a hang.

    Fails CLOSED: if the access mode cannot be read, return False and say so. A wrong 'safe' costs the
    operator a wedged install they cannot diagnose; a wrong 'unsafe' costs only the old behaviour.
    """
    if len(pvcs) > 1:
        return True, f"{len(pvcs)} distinct cache claims — independent volumes"
    if not pvcs:
        return False, "no cache claim resolved"
    pvc = next(iter(pvcs))
    rc, out, err = krun(["-n", ns, "get", "pvc", pvc, "-o", "jsonpath={.spec.accessModes}"])
    if rc != 0 or not (out or "").strip():
        return (
            False,
            f"could not read accessModes for PVC {pvc} — assuming ReadWriteOnce",
        )
    if "ReadWriteMany" in out:
        return True, f"{pvc} is ReadWriteMany"
    return (
        False,
        f"{pvc} is not ReadWriteMany (one writer node) — downloads must be sequential",
    )


def phase_c_download(
    selected: list[dict],
    prof: dict,
    dry_run: bool,
    krun=default_krun,
    tmpl_path: Path = DOWNLOAD_TMPL,
    cache_by_repo: dict[str, str] | None = None,
) -> None:
    """Render and apply one download Job per selected model.

    `cache_by_repo` maps each normalized model repository to the claim selected by `resolve_cache_claim`. A
    missing mapping is reported and skipped rather than guessed, which keeps download and mount targets equal.
    """
    ns = prof.get("NAMESPACE", "")
    cache_by_repo = cache_by_repo or {}
    failures: list[tuple[str, str]] = []  # (job_name, why) — gates the closing "All done"
    print("\n── Downloading ────────────────────────────────────────────────────")
    # Independent downloads may be applied together when their claims support safe parallel access.
    _pvc_set = {cache_by_repo.get(_norm_repo(m.get("model_repo") or ""), "") for m in selected}
    _pvc_set.discard("")
    _par_ok, _par_why = (False, "plan mode") if dry_run else _pvc_parallel_safe(ns, _pvc_set, krun)
    _pending: list[str] = []
    if len(selected) > 1:
        print(f"  {len(selected)} model(s) to fetch — " f"{'CONCURRENT' if _par_ok else 'sequential'}: {_par_why}.")
        if _par_ok:
            print("  All Jobs are applied first, then awaited, so total time is the SLOWEST model, not the sum.")
    if not dry_run:
        print(
            _stage_note(
                "Pulling the downloader image",
                "~3-8 min first time",
                "multi-GB image; cached on the node after the first pull",
            )
        )

    for model in selected:
        name = model["model_name"]
        repo = model["model_repo"]
        rev = model["model_revision"]
        rev_short = rev[:12] if rev else "?"

        # Per-recipe cache: mount THE claim the resolver chose for this model. No fallback (see docstring).
        eff_pvc = cache_by_repo.get(_norm_repo(repo), "")
        if not eff_pvc:
            print(
                f"\n  ❌ {name}: no resolved model-cache claim for {repo} — refusing to download into a "
                f"guess. (Set MODEL_CACHE_PVC or {model_cache_env_key(name)} in the cluster profile.)"
            )
            failures.append((f"llmb-download-{_model_job_slug(name)}", "no resolved cache claim"))
            continue
        eff_prof = {**prof, "MODEL_CACHE_PVC": eff_pvc}

        print(f"\n  → {name}  ({repo} @ {rev_short})  → cache {eff_pvc or '(profile default)'}")
        if not dry_run:
            _size_gib = model_size_gib(model)
            print(
                _stage_note(
                    "Downloading model to cache",
                    download_eta_text(_size_gib),
                    f"{repo}"
                    + (f", ~{_size_gib} GiB" if _size_gib else "")
                    + "; the model-cache PVC binds as this pod schedules",
                )
            )

        # Render the Job manifest.
        try:
            manifest = render_download_job(model, eff_prof, tmpl_path)
        except Exception as e:
            print(f"    ✗ Template render failed: {e}")
            continue

        # Fill cluster vars (${NAMESPACE}, ${MODEL_CACHE_PVC} → the per-recipe claim, etc.)
        manifest = _envsubst_profile(manifest, eff_prof)

        if dry_run:
            print(f"    [plan] rendered manifest ({len(manifest)} bytes):")
            for line in manifest.splitlines()[:20]:
                print(f"      {line}")
            if manifest.count("\n") > 20:
                print(f"      ... ({manifest.count(chr(10))} total lines)")
            print(f"    [plan] would apply Job to namespace '{ns}'")
            continue

        # Check for an already-running Job for this model. job_name MUST equal the template's metadata.name —
        # both derive from _model_job_slug() so they can't drift (audit #11).
        job_name = f"llmb-download-{_model_job_slug(name)}-{rev[:12]}"
        rc, out, _ = krun(["-n", ns, "get", "job", job_name, "-o", "jsonpath={.status.active}"])
        if rc == 0 and (out or "").strip() == "1":
            print(f"    ⚠ Job {job_name} is already active — skipping to avoid double-download.")
            print(f"      Monitor with: {_kubectl_hint(prof, ns)} logs -f job/{job_name}")
            continue

        # Apply the Job. (audit #1: this stdin apply bypasses krun, so pin --context from the profile here too.)
        _ctx = ["--context", prof["KUBE_CONTEXT"].strip()] if (prof.get("KUBE_CONTEXT") or "").strip() else []
        print(f"    Applying Job {job_name} ...", end=" ", flush=True)

        def _apply(_ctx: list[str] = _ctx, manifest: str = manifest):
            return subprocess.run(
                ["kubectl", *_ctx, "-n", ns, "apply", "-f", "-"],
                input=manifest,
                capture_output=True,
                text=True,
                timeout=30,
            )

        try:
            p = _apply()
        except Exception as e:
            print(f"✗\n    kubectl error: {e}")
            failures.append((job_name, str(e)[:200]))
            continue

        # A Job's spec.template is IMMUTABLE. Re-applying over a FINISHED Job of the same name — exactly
        # what happens when the download target changes (different cache claim, different revision) — is
        # rejected outright and the download silently never happens. Reconcile: delete the finished Job and
        # recreate it. The active-Job guard above already returned, so this can never kill a download in
        # flight.
        if p.returncode != 0 and "immutable" in ((p.stderr or "") + (p.stdout or "")).lower():
            print("↻", end=" ", flush=True)
            # -n ns. Every other krun on this path is namespaced; this one was not, so it targeted the
            # kubectl DEFAULT namespace and deleted nothing — and `--ignore-not-found` makes a miss exit 0,
            # so the no-op reported success. The re-apply below then hit the same immutable Job again and
            # the download silently never happened. The delete must use the selected namespace.
            _rc_del, _, _err_del = krun(
                [
                    "-n",
                    ns,
                    "delete",
                    "job",
                    job_name,
                    "--ignore-not-found",
                    "--wait=true",
                ]
            )
            if _rc_del != 0:
                print(f"✗\n    cannot replace existing Job {job_name}: {(_err_del or '').strip()[:160]}")
                failures.append((job_name, "delete-for-replace failed"))
                continue
            try:
                p = _apply()
            except Exception as e:
                print(f"✗\n    kubectl error on replace: {e}")
                failures.append((job_name, str(e)[:200]))
                continue

        if p.returncode != 0:
            print(f"✗\n    kubectl error: {p.stderr.strip()[:200]}")
            failures.append((job_name, p.stderr.strip()[:200]))
            continue
        print("✓")

        print(f"    (monitor logs: {_kubectl_hint(prof, ns)} logs -f job/{job_name})")
        if _par_ok:
            _pending.append(job_name)
            print("    queued — running in the background while the next Job is applied.")
            continue
        print("    Waiting for Job to complete (timeout 2h)...")
        _wait_for_job(ns, job_name, krun, hint=_kubectl_hint(prof, ns))

    if _pending:
        print(f"\n  All {len(_pending)} Job(s) applied and running CONCURRENTLY. Awaiting completion")
        print("  (they progress in parallel; the order below is just the reporting order):")
        for _jn in _pending:
            print(f"\n  → {_jn}")
            _wait_for_job(ns, _jn, krun, hint=_kubectl_hint(prof, ns))

    print()
    # NEVER report success over a failed apply. This printed "✓ All done" after a download Job apply had
    # failed with ✗ two lines above, so the operator moved on to `run` believing the weights were staged.
    # A failure must be summarised, and the summary must be the LAST thing on screen.
    if failures:
        print(f"  ❌ {len(failures)} model download(s) FAILED — the cache is NOT ready:")
        for _jn, _why in failures:
            print(f"     • {_jn}: {_why}")
        print("     Re-run install after resolving; `run` will CrashLoop on a missing snapshot.")
        return
    print("  ✓ All done. To benchmark:")
    print("    llmb-k8s run --recipe <recipe-path> --cluster <cluster>")


def _wait_for_job(
    ns: str,
    job_name: str,
    krun=default_krun,
    hint: str = "kubectl",
    poll_interval: int = 30,
    timeout_s: int = 7200,
) -> None:
    """Poll job status until complete or timeout."""
    start = time.time()
    last_print = 0.0
    consec_unreadable = 0  # consecutive polls where kubectl ITSELF failed (auth/network, not the Job)
    while True:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            print(f"    ✗ timeout after {int(elapsed // 60)} min")
            return

        rc, out, err = krun(
            [
                "-n",
                ns,
                "get",
                "job",
                job_name,
                "-o",
                "jsonpath={.status.succeeded},{.status.failed},{.status.active}",
            ]
        )
        # A failed status poll may be an expired local session or a transient API error; the in-cluster Job continues.
        if rc != 0:
            consec_unreadable += 1
            if consec_unreadable == 3:
                print(f"    ⚠ cannot read Job status ({(err or '').strip()[:100] or 'kubectl failed'})")
                print("      The Job runs IN-CLUSTER and is NOT affected — only this local watcher is.")
                print("      Re-authenticate to the cluster, then")
                print(f"      re-run install (idempotent), or watch it directly:  {hint} get job {job_name}")
            elif consec_unreadable > 3 and consec_unreadable % 10 == 0:
                print(
                    f"    ⚠ still unreadable after {consec_unreadable} polls ({int(elapsed // 60)} min)"
                    " — the download itself is unaffected.",
                    flush=True,
                )
            time.sleep(poll_interval)
            continue
        consec_unreadable = 0
        if rc == 0 and out:
            parts = out.strip().split(",")
            succeeded = int(parts[0] or 0) if len(parts) > 0 else 0
            failed = int(parts[1] or 0) if len(parts) > 1 else 0
            active = int(parts[2] or 0) if len(parts) > 2 else 0
            if succeeded > 0:
                print(f"    ✓ verified  ({int(elapsed // 60)} min)")
                return
            if failed > 0:
                print(f"    ✗ Job failed. Check logs: {hint} logs job/{job_name}")
                return
            if time.time() - last_print > 60:
                # "N min elapsed (active=1)" is a TIMER, not progress: it says the pod is alive and
                # nothing else, so a stalled download and a healthy one print the same line for 45
                # minutes. The downloader already reports progress on its own stdout -- surface it.
                _tail = ""
                _rcl, _outl, _ = krun(["-n", ns, "logs", f"job/{job_name}", "--tail", "1"])
                if _rcl == 0 and (_outl or "").strip():
                    _tail = "  " + " ".join((_outl or "").strip().splitlines()[-1].split())[:96]
                elif _rcl != 0:
                    _tail = "  (log tail unavailable — pod may still be pulling its image)"
                print(
                    f"    ... {int(elapsed // 60)} min elapsed  (active={active}){_tail}",
                    flush=True,
                )
                last_print = time.time()

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Phase D — per-cell bulk setup (stage → preflight → stamp)
# ---------------------------------------------------------------------------


def _run_step(argv: list[str], cwd: Path, dry_run: bool, runner=None) -> tuple[int, str]:
    """Run one setup subprocess (stage script / preflight). Returns (rc, combined_output). Injectable via
    `runner` for offline tests; dry_run short-circuits to a rendered plan line."""
    if dry_run:
        print(f"    [plan] would run: {' '.join(argv)}")
        return 0, "[plan]"
    if runner is not None:
        return runner(argv, cwd)
    try:
        p = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, str(e)


def setup_one_cell(
    cell: dict,
    cluster: str,
    prof: dict,
    *,
    dry_run: bool = False,
    state_dir: Path = STATE_DIR,
    root: Path = ROOT,
    runner=None,
) -> dict:
    """Set up ONE cell on this cluster: route via lane.py → stage script → preflight.py → emit install stamp.

    Stop at the first blocker for this cell: if dataset staging fails, print the concrete fix and skip
    preflight. Returns a result dict with the staging and preflight status.
    Pure given `runner` + injected paths (offline-testable)."""
    import lane  # local import: keeps module import cheap + matches house style

    cid = cell_id(cell)
    cell_path = cell.get("_path", "")
    cell_dir = root / cell_path
    lane_spec = lane.resolve_lane(cell.get("scenario", ""), cell.get("mode"))
    stage_script = lane_spec["stage"]
    job_mode = lane_spec.get("kind", "")

    result: dict[str, Any] = {
        "cell": cell_path,
        "id": cid,
        "status": "blocked",
        "staged": {},
        "preflight": "",
        "fix": "",
        "model_repo": "",
    }

    print(f"\n  → {cid}  ({_cell_label(cell)})")

    # --- Stage the benchmark dataset through the external llm-perf lane. ---
    stage_argv = [str(root / "scripts" / stage_script), str(cell_dir), cluster]
    print(f"    staging via {stage_script} ...", end=" ", flush=True)
    rc, out = _run_step(stage_argv, cell_dir, dry_run, runner)
    step_key = stage_script.replace(".sh", "")
    if rc != 0:
        print(GLYPH_BLOCKED)
        fix = _first_fix_line(out) or f"inspect: {' '.join(stage_argv)}"
        print(f"    {GLYPH_BLOCKED} staging failed — {fix}")
        result["staged"] = {step_key: {"ok": False}}
        result["preflight"] = "skipped"
        result["fix"] = fix
        # stamp the blocked attempt (audit trail / fleet grid); _stamp_ready() keeps it non-greyed.
        _emit_stamp(cell, cluster, result, job_mode, state_dir, dry_run)
        return result
    print(GLYPH_READY)
    result["staged"] = {step_key: {"ok": True, "sha": _sha_from_stage_output(out)}}

    # --- Preflight: preflight.py <cell> <cluster> (KUBE_CONTEXT-pinned inside preflight from the profile) ---
    print("    preflight ...", end=" ", flush=True)
    # --stage-only: install PREPARES a cell for a run LATER, so live GPU availability is not a
    # precondition — free-GPU checks report WARN (re-gated hard at run time) and the nvlink-p2p live
    # probe is skipped so staging never HOLDS GPUs. Without this a multi-GPU cell was un-installable on
    # a busy shared cluster ("no node with 8 free GPUs") even though staging needs zero GPUs.
    pf_argv = [
        sys.executable,
        str(root / "scripts" / "preflight.py"),
        str(cell_dir),
        cluster,
        "--stage-only",
    ]
    rc, out = _run_step(pf_argv, cell_dir, dry_run, runner)
    if dry_run:
        verdict = "pass"
    elif rc == 0:
        verdict = "warn" if "WARN" in out else "pass"
    else:
        verdict = "fail"
    result["preflight"] = verdict
    if verdict == "fail":
        print(GLYPH_BLOCKED)
        fix = _first_fix_line(out) or f"re-run: {' '.join(pf_argv[1:])}"
        print(f"    {GLYPH_BLOCKED} preflight FAILED — {fix}")
        result["status"] = "blocked"
        result["fix"] = fix
    else:
        print(GLYPH_READY + (f"  ({verdict})" if verdict == "warn" else ""))
        result["status"] = "ready"

    _emit_stamp(cell, cluster, result, job_mode, state_dir, dry_run)
    return result


def _first_fix_line(output: str) -> str:
    """Extract the first concrete fix hint from a stage/preflight subprocess output. preflight prints
    '→ fix: <cmd>' lines; stage scripts print an error to stderr. Returns '' if none found. Pure.
    """
    for ln in (output or "").splitlines():
        s = ln.strip()
        if "→ fix:" in s:
            return s.split("→ fix:", 1)[1].strip()
    # fall back to the first non-empty error-ish line
    for ln in (output or "").splitlines():
        s = ln.strip()
        if s and (":" in s):
            return s
    return ""


def _sha_from_stage_output(output: str) -> str:
    """Best-effort pull of a staged sha256 from stage-script output (for the stamp's audit trail). Pure."""
    m = re.search(r"\b([0-9a-f]{64})\b", output or "")
    return m.group(1) if m else ""


def _emit_stamp(cell, cluster, result, job_mode, state_dir, dry_run) -> None:
    if dry_run:
        print(
            f"    [plan] would stamp {cell_id(cell)} → {result['status']} "
            f"(preflight={result['preflight'] or 'n/a'})"
        )
        return
    write_install_stamp(
        cluster=cluster,
        cell_path=cell.get("_path", ""),
        recipe_hash=cell.get("recipe_hash", ""),
        model_repo=result.get("model_repo", ""),
        staged=result.get("staged", {}),
        preflight=result.get("preflight", ""),
        job_mode=job_mode,
        state_dir=state_dir,
    )


def phase_d_setup(
    selected: list[dict],
    cluster: str,
    prof: dict,
    dry_run: bool,
    *,
    state_dir: Path = STATE_DIR,
    root: Path = ROOT,
    runner=None,
    misplaced: dict | None = None,
) -> list[dict]:
    """Per-cell bulk setup for every selected cell. Prints per-cell progress + a summary. Returns the list of
    per-cell result dicts.

    `misplaced` maps model -> (other_claim, why) for models whose weights are STAMPED on a claim other than
    the one they resolve to. Such a cell is downgraded to needs-input: READY must not be reachable while a
    model points at a claim that demonstrably holds someone else's weights, because the next step printed
    below is `llmb-k8s run`, which would deploy a server that cannot find its model. The check already
    existed (find_misplaced_weights) but only printed during the cache phase, where it did not reach the
    verdict — a live plan reported `✓ ready: 31  ❌ blocked: 0` for exactly that configuration.
    """
    print("\n── Per-cell setup ─────────────────────────────────────────────────")
    misplaced = misplaced or {}
    results = []
    for cell in selected:
        # thread the union model repo into the stamp (informational): first model this cell references.
        r = setup_one_cell(
            cell,
            cluster,
            prof,
            dry_run=dry_run,
            state_dir=state_dir,
            root=root,
            runner=runner,
        )
        _mis = misplaced.get(cell.get("model") or "")
        if _mis and r["status"] == "ready":
            _other, _mwhy = _mis
            _key = model_cache_env_key(cell.get("model") or "")
            r["status"] = "needs-input"
            r["fix"] = (
                f"weights for '{cell.get('model')}' are on '{_other}', not the claim this cell "
                f"resolves to ({_mwhy}). Run `llmb-k8s install {cluster} --adopt-cache`, or set "
                f'{_key}="{_other}" in cluster-profiles/{cluster}.env'
            )
        results.append(r)

    # summary
    ready = [r for r in results if r["status"] == "ready"]
    needs = [r for r in results if r["status"] == "needs-input"]
    blocked = [r for r in results if r["status"] == "blocked"]
    print("\n── Summary ────────────────────────────────────────────────────────")
    print(
        f"  {GLYPH_READY} ready: {len(ready)}   "
        f"{GLYPH_NEEDS} needs-input: {len(needs)}   "
        f"{GLYPH_BLOCKED} blocked: {len(blocked)}"
    )
    for r in needs:
        print(f"  {GLYPH_NEEDS} {r['id']}: {r['fix']}")
    for r in blocked:
        print(f"  {GLYPH_BLOCKED} {r['id']}: {r['fix']}")
    if ready:
        # Emit the recipe PATH (r["cell"]), not the bare slug (r["id"]): `llmb-k8s run` requires
        # <cell-dir>/recipe.yaml and errors "no recipe.yaml in <slug>" on a bare id — so this must be a
        # copy-paste-runnable command (Bug C). Fall back to the id only if a path is somehow absent.
        _next = ready[0].get("cell") or ready[0]["id"]
        print(f"\n  Next: llmb-k8s run {_next} {cluster}")
    if needs:
        # Point at the REMEDY, not at `run`. If any cell needs a cache re-point, --adopt-cache is the one
        # command that fixes all of them at once, and it is the thing an operator will not think to run.
        if any("--adopt-cache" in (r.get("fix") or "") for r in needs):
            print(f"\n  Fix the cache pointing first:  llmb-k8s install {cluster} --adopt-cache")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _argv = sys.argv[1:] if argv is None else argv
    # Thin sub-entry the `run` inline-stage path shells out to: record a stage attempt into the per-cluster
    # install stamp (so a cell staged by `run`, not just `install`, appears in fleet's INSTALLED inventory).
    if _argv and _argv[0] == "--record-stage":
        return _cli_record_stage(_argv[1:])
    # The cache-claim resolver every shell consumer calls (scripts/_model_cache.sh). Kept as a sub-entry so
    # deploy.sh/sweep.sh/stage-*.sh get the SAME resolve_cache_claim() the download path uses.
    if _argv and _argv[0] == "--resolve-cache":
        return _cli_resolve_cache(_argv[1:])
    parser = argparse.ArgumentParser(
        description="install.py — provision cluster prereqs + download models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              scripts/install.py example-gpu-cluster
              scripts/install.py --cluster example-gpu-cluster
              scripts/install.py example-gpu-cluster --dry-run
        """),
    )
    parser.add_argument(
        "cluster",
        nargs="?",
        help="Cluster profile name (e.g. example-gpu-cluster).  Positional or --cluster.",
    )
    parser.add_argument(
        "--cluster",
        dest="cluster_flag",
        metavar="CLUSTER",
        help="Named flag alias for the cluster argument.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="OFFLINE plan: render + print the plan with NO cluster probing and NO changes.",
    )
    mode.add_argument(
        "--live-plan",
        action="store_true",
        help="LIVE plan: probe real cluster state (namespace, secrets, PVC free space, on-PVC "
        "models) and print what WOULD be installed — but apply nothing (R2-3/#11).",
    )
    parser.add_argument(
        "--skip-pvc-probe",
        action="store_true",
        help="Skip PVC free-space and on-PVC model probes (faster, no mounter pods). "
        "NOTE: free space then reads as UNKNOWN, and a large download is REFUSED "
        "unless you also pass --allow-unmeasured-download.",
    )
    parser.add_argument(
        "--allow-unmeasured-download",
        dest="allow_unmeasured_download",
        action="store_true",
        default=_env_flag("LLMB_ALLOW_UNMEASURED_DOWNLOAD"),
        help="Proceed with a large download even when the cache's free space is UNKNOWN. "
        "For clusters where restricted PodSecurity/Kyverno blocks the mounter pod the "
        "probe needs (free space is then permanently unknowable). Env: "
        "LLMB_ALLOW_UNMEASURED_DOWNLOAD=1 (only 1/true/yes/on enable it).",
    )
    parser.add_argument(
        "--no-allow-unmeasured-download",
        dest="allow_unmeasured_download",
        action="store_false",
        help="Force the refusal back on, overriding LLMB_ALLOW_UNMEASURED_DOWNLOAD from the "
        "environment. Without this there is no way to say NO on the command line.",
    )
    # Headless flags (mirror the SLURM express recipe multi-select). Each stage/preflight call is
    # KUBE_CONTEXT-pinned and the stamp is keyed by cluster, so these are safe for concurrent multi-cluster use.
    parser.add_argument(
        "--recipes",
        metavar="CELL[,CELL...]",
        help="Headless: set up these cells (match on catalog name, full cell path, or path basename). "
        "Skips the interactive recipe picker.",
    )
    parser.add_argument(
        "--all-matching",
        action="store_true",
        help="Headless: set up EVERY GPU-matching, not-yet-installed cell and AUTO-DOWNLOAD "
        "the union of models they need (no download prompt).",
    )
    parser.add_argument(
        "--list-recipes",
        action="store_true",
        help="Print the GPU-compat matrix + which cells are already installed on this "
        "cluster, then exit. Offline — no cluster probing.",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Stage + preflight only: skip the Phase C model-cache download of the union.",
    )
    parser.add_argument(
        "--from-init",
        action="store_true",
        help="Invoked by `llmb-k8s init` right after it proved the cluster RUN-READY. "
        "Skips the redundant Phase A profile-review prompt and drops straight onto the "
        "recipe selector. Interactive selection is unchanged.",
    )

    parser.add_argument(
        "--adopt-cache",
        action="store_true",
        help="When no model-cache PVC is configured for a selected recipe, pick the "
        "best-fitting EXISTING claim on the cluster and WRITE it into "
        "cluster-profiles/<cluster>.env (MODEL_CACHE_PVC / MODEL_CACHE_PVC_<MODEL>), "
        "then continue. Discovery as profile authoring — never a runtime override.",
    )

    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Re-run setup for cells that are ALREADY installed (default: they are a no-op). "
        "Never re-downloads a model the cache already vouches for.",
    )

    args = parser.parse_args(argv)

    # No cluster given → ask (Phase 0). `install <cluster>` is untouched; this is purely the extra door.
    cluster, _err = resolve_cluster_arg(
        args.cluster_flag or args.cluster,
        is_tty=sys.stdin.isatty(),
        profiles=_pr.list_profiles(),
    )
    if _err:
        print(_err)
        return 2

    dry_run = args.dry_run
    live_plan = args.live_plan
    plan_only = dry_run or live_plan  # neither mode applies changes / prompts
    # Headless recipe multi-select: no interactive pickers, auto-confirm the profile, auto-download the union.
    headless = bool(args.recipes) or args.all_matching
    # --from-init: init already ran the readiness battery and proved the profile RUN-READY, so the Phase A
    # profile-review prompt is redundant. Auto-confirm it and land directly on the interactive recipe selector.
    from_init = args.from_init
    do_probe = live_plan or (not dry_run and not args.skip_pvc_probe)  # live-plan always probes
    if dry_run:
        print(
            "  [dry-run mode] — OFFLINE plan: no cluster probing, no changes " "(use --live-plan to probe real state)\n"
        )
    elif live_plan:
        print("  [live-plan mode] — probing REAL cluster state; nothing will be applied\n")

    # ── Resolve profile ──────────────────────────────────────────────────
    # A dry-run/list operation promises to be offline. Validate that the named profile exists,
    # but do not turn an expired VPN/SSO session into a planning failure.
    offline_profile_only = dry_run or args.list_recipes
    resolution = _pr.resolve(
        cluster,
        profiles_dir=_pr.PROFILES_DIR,
        probe=(lambda _ctx: True) if offline_profile_only else _pr.default_probe,
    )
    if not resolution.ok:
        print(resolution.message)
        return resolution.exit_code

    prof_path = _pr.profile_env_path(cluster)
    prof = _pr._read_env(prof_path)
    ns = prof.get("NAMESPACE", "")
    # audit #1 — pin every kubectl to the profile's cluster (not the global current-context).
    krun = make_krun(prof.get("KUBE_CONTEXT", "").strip())

    # ── Catalog + GPU-matching cells + this cluster's install stamp ───────
    catalog = load_catalog()
    if not catalog:
        print(f"\n  ⚠ No catalog found at {CATALOG_PATH}. Run build_catalog.py first.")
        return 1
    cells = gpu_matching_cells(catalog, prof)
    stamps = read_install_stamps(cluster)

    # --list-recipes: print the GPU-compat matrix + already-installed set, then exit (offline, no probing).
    if args.list_recipes:
        list_recipes(cells, stamps, cluster, _pr.profile_gpu_type(prof))
        return 0

    if not cells:
        print(
            f"\n  ⚠ No catalog cells target this cluster's GPU "
            f"({_pr.profile_gpu_type(prof) or '?'}). Nothing to install here."
        )
        return 0

    # ── Phase A — profile review ─────────────────────────────────────────
    if not do_probe:
        state: dict = {
            "ns_ok": None,
            "gpu_nodes_total": None,
            "gpu_nodes_used": None,
            "pull_secret_ok": None,
            "hf_secret_ok": None,
            "pvc_phase": None,
            "pvc_exists": None,
            "pvc_free_gib": None,
        }
    else:
        print("  Probing cluster state...")
        state = probe_cluster_state(prof, krun=krun)
        # Also get PVC free space while we're at it (reuse the probe_pvc_free_gib call)
        pvc = prof.get("MODEL_CACHE_PVC", "")
        if pvc and ns and state.get("pvc_phase") == "Bound":
            print("  Probing PVC free space...")
            free_gib = probe_pvc_free_gib(ns, pvc, krun=krun, prof=prof)
            state["pvc_free_gib"] = free_gib

    # Headless auto-confirms the profile (no tty prompt); interactive/plan modes go through the panel.
    # --from-init also auto-confirms: init just proved this profile RUN-READY, so re-asking "Proceed with
    # this profile?" would be a redundant gate before the recipe selector (still shows the review panel).
    proceed = phase_a_review(cluster, prof, resolution, state, plan_only or headless or from_init)
    if not proceed:
        print("  Aborted.")
        return 0

    # ── Ensure prerequisites: namespace + model-cache PVC + secrets (G9/G10/G3) ──
    # Idempotent + portable + runs on EVERY path (headless copy/paste included), so a fresh cluster no longer
    # needs a hand-run `kubectl create namespace` / `kubectl apply` PVC / `kubectl create secret`. Ordered
    # ns → PVC → secrets so the later download Job (which mounts the PVC into the ns) has both. plan modes
    # (dry-run/live-plan) print the plan and apply nothing. Interactive attaches a getpass fallback when a
    # secret has no env/cred-file/profile source; headless resolves values from HF_TOKEN / NGC_API_KEY / the
    # user's own ~/.cache/huggingface/token · ~/.ngc/config / the profile / ~/.config/llmb/secrets and, if a
    # value is genuinely absent, prints the EXACT key to set (never a silent skip).
    interactive_secrets = (not plan_only) and (not headless) and sys.stdin.isatty()
    apply_mf = make_manifest_applier(prof.get("KUBE_CONTEXT", "").strip())
    ensure_prerequisites(
        prof,
        state,
        cluster,
        krun=krun,
        applier=apply_mf,
        plan_only=plan_only,
        interactive=interactive_secrets,
        probe=not dry_run,
    )

    # ── What is ALREADY installed here (cluster truth, before you choose) ─
    # Read-only. Uses fleet's own honest verdicts so install and fleet can never disagree; a cluster we can't
    # read says nothing rather than "absent".
    live = {
        "caches": {},
        "cells": {},
        "readable": False,
        "skip_reason": "plan mode — the cluster was not probed" if not do_probe else "",
    }
    if do_probe:
        live = probe_cluster_install_state(prof, krun=krun)
    print(render_installed_panel(cells, stamps, prof, live, cluster))

    # ── Phase B — recipe / cell multi-select ─────────────────────────────
    if headless:
        selected_cells, sel_errors = resolve_recipe_selection(
            cells, stamps, args.recipes, args.all_matching, reinstall=args.reinstall
        )
        for e in sel_errors:
            print(f"  ❌ {e}")
        if sel_errors and not selected_cells:
            available = ", ".join(cell_id(c) for c in cells if not cell_installed(c, stamps)) or "(none)"
            print(f"     available: {available}")
            return 2
        if selected_cells:
            print("\n── Recipes selected (headless) ────────────────────────────────────")
            for c in selected_cells:
                print(f"  • {cell_id(c)}   [{c.get('status', '?')}]  {_cell_label(c)}")
    else:
        selected_cells = phase_b_recipe_selection(cells, stamps, plan_only)

    if not selected_cells:
        print("\n  Nothing to set up.")
        return 0

    # ── Ensure per-recipe model-cache PVCs (one correct claim per selected recipe) ──
    # Runs AFTER selection (needs the chosen cells) and BEFORE Phase C download (the download Job mounts the
    # claim, so it must exist first). Idempotent + never clobbers a Bound claim. Dedups by name so recipes
    # sharing a model share one cache. applier is context-pinned like every other apply on this path.
    applier = make_manifest_applier(prof.get("KUBE_CONTEXT", "").strip())

    # Validate cache routing before provisioning or downloading.
    _cache_errs = validate_cache_config(selected_cells, prof, cluster)
    # `--adopt-cache` persists explicit operator-approved mappings in the profile. Plan modes report the
    # changes without writing them; offline dry-run cannot inspect live PVCs.
    if args.adopt_cache and plan_only:
        print("\n── Model cache: --adopt-cache (plan) ──────────────────────────────")
        if dry_run:
            print(
                "  --adopt-cache reads the cluster's PVCs to find which claim already holds each model, "
                "so it cannot run under --dry-run (offline)."
            )
            print("  Re-run with --live-plan to see the exact profile lines it would write.")
        else:
            _cands_p, _perr_p = list_cache_candidates(prof.get("NAMESPACE", ""), krun)
            if _perr_p:
                print(f"  ⚠ cannot list PVCs in '{prof.get('NAMESPACE', '')}': {_perr_p}")
                print("    This is 'could not look', not 'nothing there' — no adoption can be planned.")
            else:
                _keys_p, _notes_p = plan_cache_adoption(selected_cells, _cands_p, prof)
                for _n in _notes_p:
                    print(f"  · {_n}")
                for _k, _v in sorted(_keys_p.items()):
                    print(f'  [plan] would write to cluster-profiles/{cluster}.env: {_k}="{_v}"')
                if not _keys_p:
                    print(
                        "  [plan] no profile change needed — every selected model already resolves to a "
                        "claim that fits it"
                    )
                else:
                    # Plan the REST of this run against what adoption would produce, so the readiness
                    # verdict below describes the post-adoption world instead of failing on a gap the
                    # operator just asked to close.
                    prof = {**prof, **_keys_p}
                    _cache_errs = validate_cache_config(selected_cells, prof, cluster)
                    print("  [plan] the rest of this plan assumes those lines are in place.")
    if args.adopt_cache and not plan_only:
        _cands, _perr = list_cache_candidates(prof.get("NAMESPACE", ""), krun)
        if _perr:
            print(f"\n  ❌ --adopt-cache cannot list PVCs in '{prof.get('NAMESPACE', '')}': {_perr}")
            print("     This is 'could not look', not 'nothing there' — refusing to adopt on a blind guess.")
            return 2
        _keys, _notes = plan_cache_adoption(selected_cells, _cands, prof)
        print("\n── Model cache: adopting into the profile ─────────────────────────")
        for n in _notes:
            print(f"  · {n}")
        if _keys:
            _envp = _pr.profile_env_path(cluster)
            for ln in apply_profile_keys(_envp, _keys):
                print(f"  + {_envp.name}: {ln}")
            prof.update(_keys)
        else:
            print("  · no profile change needed — every selected model already resolves to a claim that " "fits it")
        _cache_errs = validate_cache_config(selected_cells, prof, cluster)
    if _cache_errs:
        print("\n── Model cache: NOT CONFIGURED ────────────────────────────────────")
        for e in _cache_errs:
            print(f"  ❌ {e}")
        # ADVISORY discovery: show what IS on the cluster and the exact lines to paste, plus the
        # --adopt-cache one-liner that writes them. Never auto-adopt without the flag.
        if not dry_run:
            _cands, _perr = list_cache_candidates(prof.get("NAMESPACE", ""), krun)
            print()
            print(render_cache_candidates_advice(_cands, _perr, selected_cells, cluster))
        print("\n  Nothing was downloaded and nothing was applied. Re-run install after editing the profile.")
        return 2
    # Report when another stamped claim already contains the selected model. This remains advisory because
    # only the operator can decide whether to adopt or replace that cache.
    _misplaced: dict = {}
    if not dry_run:
        _cands_r, _perr_r = list_cache_candidates(prof.get("NAMESPACE", ""), krun)
        if not _perr_r:
            _seen_mis: set = set()
            for _c in selected_cells:
                _model = _c.get("model") or ""
                _claim = resolve_cache_claim(_c, prof)[0]
                if not _model or _model in _seen_mis:
                    continue
                _other, _why_mis = _mc.find_misplaced_weights(_model, _claim, _cands_r, _cell_model_revision(_c) or "")
                if _other:
                    _seen_mis.add(_model)
                    _misplaced[_model] = (_other, _why_mis)
                    print(f"\n  ⚠ model '{_model}' resolves to '{_claim}', but {_why_mis}.")
                    print("     If those are the weights you want, point this model at that claim:")
                    print(f'       {model_cache_env_key(_model)}="{_other}"   ' f"in cluster-profiles/{cluster}.env")
                    print(
                        f"     Otherwise install will download it into '{_claim}' — which SUCCEEDS if "
                        f"there is room, so nothing will fail to tell you."
                    )
    _claim_of = {cell_id(c): resolve_cache_claim(c, prof)[0] for c in selected_cells}
    print("\n── Model cache: resolved claims ───────────────────────────────────")
    for _claim in sorted(set(_claim_of.values())):
        _n = sum(1 for v in _claim_of.values() if v == _claim)
        print(f"  ✓ {_claim}   ({_n} cell(s))")
    print(
        "    Same claim for the download Job AND the server mount — both read " "${MODEL_CACHE_PVC} from this profile."
    )

    _cache_results = ensure_recipe_cache_pvcs(
        selected_cells,
        prof,
        krun=krun,
        applier=applier,
        plan_only=plan_only,
        probe=not dry_run,
    )
    _cache_blocking = {
        "failed",
        "attach-failed",
        "undersized",
        "access-unknown",
        "access-mismatch",
    }
    if any(st in _cache_blocking for st, _msg, _spec in _cache_results):
        print("\n  Nothing was downloaded. Fix the model-cache error above and re-run install.")
        return 2
    # Resolve every model download target; refuse unresolved mappings rather than falling back.
    cache_by_repo, _cbr_errs = cache_pvc_by_repo(selected_cells, {**prof, "CLUSTER": cluster})
    if _cbr_errs:
        print("\n── Model cache: cannot determine a download target ────────────────")
        for e in _cbr_errs:
            print(f"  ❌ {e}")
        print("\n  Nothing was downloaded. Fix the above and re-run.")
        return 2

    # ── Phase C — model-cache (union of the selected cells' models) ───────
    models = models_for_cells(selected_cells)
    if not models:
        print("\n  ⚠ Selected cells reference no downloadable model (no serving.model_repo). Skipping download.")
    elif args.skip_model_download:
        print("\n── Model cache ────────────────────────────────────────────────────")
        print(f"  [--skip-model-download] skipping the union of {len(models)} model(s); " f"stage + preflight only.")
    else:
        if headless:
            # Auto-download the whole union (owner chose auto-download; no --yes-download gate). Refuse
            # unpinned-revision models the same way the interactive picker does.
            print("\n── Model cache (union — auto-download) ─────────────────────────────")
            dl_models = []
            for m in models:
                if m.get("model_revision"):
                    dl_models.append(m)
                    print(f"  • {m['model_name']}  ({m['model_repo']} @ {m['model_revision'][:12]})")
                else:
                    print(
                        f"  ⚠  {m['model_name']}: no pinned model_revision — skipping "
                        f"(pin serving.model_revision in recipe.yaml first)"
                    )
            # Probe existing cache contents before scheduling downloads. A completion sentinel alone is not
            # sufficient when stronger shard evidence reports an incomplete snapshot.
            if do_probe and ns and dl_models:
                _subpath = (prof.get("MODEL_CACHE_SUBPATH") or ".").strip() or "."
                print(f"\n  Checking what is already on the cache — {len(dl_models)} model(s).")
                _keep = []
                for m in dl_models:
                    _claim = cache_by_repo.get(_norm_repo(m["model_repo"]), "")
                    if not _claim:
                        _keep.append(m)
                        continue
                    _w: list = []
                    _facts: dict = {}
                    _st = probe_model_on_pvc(
                        ns,
                        _claim,
                        m["model_repo"],
                        m["model_revision"],
                        _subpath,
                        krun=krun,
                        prof=prof,
                        why=_w,
                        facts_out=_facts,
                    )
                    _detail = _w[0][1] if _w else ""
                    print(f"    {m['model_name']}: {_st}" + (f" — {_detail}" if _detail else ""))
                    if _st == _mc.STATE_COMPLETE:
                        # VERIFY-AND-STAMP, then skip. This is the acceptance bar: a model proven complete
                        # is never re-downloaded, and the proof is recorded so the next run is instant.
                        # The SKIP follows the verdict; the STAMP follows the stronger sentinel_worthy bar,
                        # because a sentinel is permanent and a verdict is not (see stamp_download_sentinel).
                        _worthy, _wwhy = _mc.sentinel_worthy(_facts)
                        if plan_only:
                            print("      ✓ complete — would skip the download")
                        elif _worthy:
                            _ok, _sm = stamp_download_sentinel(
                                ns,
                                _claim,
                                m["model_repo"],
                                m["model_revision"],
                                _subpath,
                                krun=krun,
                                prof=prof,
                            )
                            print(
                                f"      ✓ complete — skipping download"
                                f" ({'sentinel written' if _ok else 'sentinel not written: ' + _sm})"
                            )
                        else:
                            print(f"      ✓ complete — skipping download (no sentinel written: {_wwhy})")
                        continue
                    if _st == _mc.STATE_UNKNOWN:
                        print("      ⚠ could not verify — treating as NOT present, so nothing is skipped")
                    _keep.append(m)
                dl_models = _keep
                if not dl_models:
                    print("  Nothing to download — every model is already complete on its claim.")
            # CAPACITY GUARD — per RESOLVED claim, shared with the interactive door below.
            dl_models, _ = capacity_gate(
                dl_models,
                cache_by_repo,
                ns,
                krun=krun,
                do_probe=do_probe,
                plan_only=plan_only,
                prof=prof,
                allow_unmeasured=args.allow_unmeasured_download,
            )
            if dl_models:
                phase_c_download(
                    dl_models,
                    prof,
                    plan_only,
                    krun=krun,
                    tmpl_path=DOWNLOAD_TMPL,
                    cache_by_repo=cache_by_repo,
                )
        else:
            # Interactive: reuse the existing model-selection widget over the UNION, then download.
            on_pvc_status: dict[str, str] = {}
            if do_probe and ns:
                # Each probe SCHEDULES A SHORT-LIVED POD that mounts the cache PVC and looks for the
                # download sentinel -- so it costs pod scheduling + image pull + (on a
                # WaitForFirstConsumer class) the PVC's first bind. That is tens of seconds PER MODEL,
                # and it used to print one line and then go silent for minutes. Narrate per model:
                # an operator must never have to guess whether this is working or wedged.
                subpath = (prof.get("MODEL_CACHE_SUBPATH") or ".").strip() or "."
                _probe = [m for m in models if m.get("model_revision")]
                print(f"\n  Probing on-PVC model status — {len(_probe)} model(s) to check.")
                print("    Each check runs a short-lived pod that mounts the cache PVC (~20-60s each;")
                print("    longer on the first one if the PVC still has to bind). Nothing is downloaded here.")
                for _i, m in enumerate(_probe, 1):
                    _nm = m.get("model_name") or _norm_repo(m["model_repo"])
                    print(f"    [{_i}/{len(_probe)}] {_nm} ... ", end="", flush=True)
                    _t0 = time.time()
                    # Probe the claim resolved for this model, including any per-model profile override.
                    _mpvc = cache_by_repo.get(_norm_repo(m["model_repo"]), "")
                    if not _mpvc:
                        on_pvc_status[_norm_repo(m["model_repo"])] = _mc.STATE_UNKNOWN
                        print("no claim resolved — skipped", flush=True)
                        continue
                    _why: list = []
                    _pfacts: dict = {}
                    _st = probe_model_on_pvc(
                        ns,
                        _mpvc,
                        m["model_repo"],
                        m["model_revision"],
                        subpath,
                        krun=krun,
                        prof=prof,
                        why=_why,
                        facts_out=_pfacts,
                    )
                    # VERIFY-AND-STAMP. The probe just PROVED this snapshot complete (every shard the
                    # index names resolves, no .incomplete blobs). The only thing missing was the
                    # sentinel — nemotron-ultra-nvfp4-cache holds 113/113 safetensors and has no
                    # .llmb_download_done directory at all, because it was staged before that convention
                    # existed. Write the sentinel now, so this costs a full re-verification ONCE instead
                    # of every install, and so preflight and fleet get the same instant answer.
                    if _st == _mc.STATE_COMPLETE and not plan_only:
                        _worthy, _wwhy = _mc.sentinel_worthy(_pfacts)
                        if _worthy:
                            _facts_ok, _smsg = stamp_download_sentinel(
                                ns,
                                _mpvc,
                                m["model_repo"],
                                m["model_revision"],
                                subpath,
                                krun=krun,
                                prof=prof,
                            )
                            _why.append(("stamped" if _facts_ok else "stamp-skipped", _smsg, ""))
                        else:
                            _why.append(
                                (
                                    "stamp-withheld",
                                    _wwhy,
                                    "a sentinel is permanent — it is written only on the strong proof",
                                )
                            )
                    on_pvc_status[_norm_repo(m["model_repo"])] = _st
                    print(f"{_st}  ({int(time.time() - _t0)}s)", flush=True)
                    for _code, _reason, _hint in _why:
                        print(f"        {_code}: {_reason}", flush=True)
                        if _hint:
                            print(f"        → {_hint}", flush=True)
            # free_gib/pvc are per-CLAIM now, so the widget gets no single global number to mislead with;
            # the authoritative check is capacity_gate below, which measures each claim the models land in.
            dl_models = phase_b_model_selection(models, None, "", on_pvc_status, plan_only)
            if dl_models:
                dl_models, _ = capacity_gate(
                    dl_models,
                    cache_by_repo,
                    ns,
                    krun=krun,
                    do_probe=do_probe,
                    plan_only=plan_only,
                    prof=prof,
                    allow_unmeasured=args.allow_unmeasured_download,
                )
            if dl_models:
                phase_c_download(
                    dl_models,
                    prof,
                    plan_only,
                    krun=krun,
                    tmpl_path=DOWNLOAD_TMPL,
                    cache_by_repo=cache_by_repo,
                )

    # ── Phase D — per-cell bulk setup (stage → preflight → stamp) ─────────
    # plan modes (dry-run AND live-plan) must apply nothing, so gate staging/preflight on plan_only.
    phase_d_setup(selected_cells, cluster, prof, plan_only, misplaced=_misplaced)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
