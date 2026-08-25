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

"""k8s_config.py — XDG config-storage managers for the `llmb-k8s init` wizard (the wizard contract §5).

Two small persisted artifacts, mirroring the SLURM installer's storage conventions
(`cli/llmb-install/src/llmb_install/config/system.py`) but k8s-native and dependency-light:

  ~/.config/llmb/k8s-system.yaml     — stable, CLUSTER-AGNOSTIC operator prefs reused across clusters
                                       (owner, CONNECT_CMD pattern, storage-class name hints). Feeds
                                       --express. Cluster-SPECIFIC facts are NEVER persisted here — they
                                       are re-detected every run (Q5).
  ~/.config/llmb/k8s-init-state.yaml — resume state for a wizard interrupted mid-collect, with the SAME
                                       7-day expiry + `datetime.fromisoformat` staleness check as SLURM's
                                       InstallStateManager.load_install_state.

Every write is the atomic tmp-file → chmod(0600)-on-tmp → replace() pattern (verified `system.py:106-109`,
`system.py:250-253`) — never world-readable at the process umask, never observed half-written. The profile
itself (the working artifact) is written by profile_init.write_profile with the same discipline; these two
files are the wizard's side-band state.

Pure enough to unit-test: pass an explicit `path=` (the selftest points it at a tmp dir); the live default
is the XDG path. No cluster, no network.
"""

from __future__ import annotations

import os
import stat
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dep of the toolchain
    yaml = None  # type: ignore

STATE_MAX_AGE_DAYS = 7


def _atomic_write(path: Path, render: Callable) -> None:
    """Atomic + 0600 write via a UNIQUE temp file in the same dir, then atomic replace().

    PARALLEL-SAFETY: the two side-band files (k8s-init-state.yaml, k8s-system.yaml) are SHARED across
    concurrent inits for different clusters. A fixed `.{name}.tmp` would let two writers collide on the same
    temp path and tear each other's write. `tempfile.mkstemp` gives every writer a unique name AND opens it
    O_WRONLY|O_CREAT|O_EXCL with mode 0600 — so there is never a world-readable window at the process umask
    (the old write_text()-then-chmod pattern's flaw), and never a torn shared file. `render(fh)` writes the
    body; replace() is a same-filesystem atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmpname = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmpname)
    try:
        with os.fdopen(fd, "w") as f:
            render(f)
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 0600 (mkstemp already 0600; explicit + belt-and-suspenders)
        tmp.replace(path)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def config_dir() -> Path:
    """XDG-compliant config dir: $XDG_CONFIG_HOME/llmb or ~/.config/llmb (mirrors _get_system_config_dir())."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return (Path(xdg) / "llmb") if xdg else (Path.home() / ".config" / "llmb")


def system_config_path() -> Path:
    return config_dir() / "k8s-system.yaml"


def init_state_path() -> Path:
    return config_dir() / "k8s-init-state.yaml"


def atomic_write_yaml(path: Path, data: dict) -> None:
    """Write `data` as YAML atomically and 0600 (the wizard contract §5/M2). tmp in the SAME dir (so replace() is
    a same-filesystem atomic rename), chmod the tmp BEFORE it carries the final name, then replace().
    """
    if yaml is None:
        raise RuntimeError("pyyaml required for k8s config storage")
    _atomic_write(
        path,
        lambda f: yaml.safe_dump(data, f, default_flow_style=False, indent=2, sort_keys=True),
    )


def _read_yaml(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.exists() or yaml is None:
        return None
    try:
        data = yaml.safe_load(path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


# ─────────────────────────────────────────────────────────────────────────────
# System config — stable cluster-agnostic prefs (feeds --express)
# ─────────────────────────────────────────────────────────────────────────────

# The ONLY keys persisted cross-cluster (Q5). Everything cluster-specific (context/ns/PVC/secrets/IPs) is
# re-detected every run and is NOT stored here. SC name hints are ranked HINTS that still require confirm.
SYSTEM_PREF_KEYS = ("owner", "connect_cmd", "artifacts_sc_hint", "control_sc_hint")


def load_system_config(path: Optional[Path] = None) -> dict:
    """Load stable operator prefs (or {} if none). Never raises."""
    data = _read_yaml(path or system_config_path())
    if not data:
        return {}
    return {k: v for k, v in data.items() if k in SYSTEM_PREF_KEYS and v not in (None, "")}


def save_system_config(prefs: dict, path: Optional[Path] = None) -> None:
    """READ-MODIFY-WRITE persist of ONLY the cluster-agnostic pref keys (Q5), under atomic replace. Merges
    onto any existing prefs (so a second cluster's init doesn't wipe the first's owner/hints) and silently
    drops anything cluster-specific — a per-cluster fact can never leak into this cross-cluster file.
    """
    p = path or system_config_path()
    merged = load_system_config(p)  # existing cluster-agnostic prefs
    for k in SYSTEM_PREF_KEYS:
        if str(prefs.get(k) or "").strip():
            merged[k] = prefs[k]
    atomic_write_yaml(p, merged)


# ─────────────────────────────────────────────────────────────────────────────
# Init resume state — PER-CLUSTER-KEYED, 7-day expiry (mirrors InstallStateManager)
#
# CONCURRENCY (coordinator requirement): the wizard runs for MANY clusters in parallel, so the single
# state file is a top-level `clusters: {<name>: {answers, timestamp}}` MAP keyed by cluster name — a
# concurrent `init --cluster Y` never stomps `init --cluster X`'s entry. Every write is read-modify-write
# under the same atomic tmp+replace (the file is never torn; last-writer-wins on the map, and each writer
# only ever touches its OWN cluster key). No global lockfile — parallel inits across clusters are never
# serialized. Resume state is best-effort (7-day, non-critical), so the residual RMW lost-update window
# (two writers overlapping to the exact same byte) at worst drops another cluster's resume hint, never the
# profile artifact (which is the only cluster-specific write target, already per-cluster).
# ─────────────────────────────────────────────────────────────────────────────


def _load_state_map(path: Path) -> dict:
    data = _read_yaml(path) or {}
    clusters = data.get("clusters")
    return clusters if isinstance(clusters, dict) else {}


def save_init_state(cluster: str, answers: dict, path: Optional[Path] = None) -> None:
    """Persist an interrupted wizard's collected answers under the per-cluster key so a re-run can resume.
    Read-modify-write under atomic replace — never clobbers another cluster's entry."""
    p = path or init_state_path()
    clusters = _load_state_map(p)
    clusters[cluster] = {"answers": answers, "timestamp": datetime.now().isoformat()}
    atomic_write_yaml(p, {"clusters": clusters})


def load_init_state(
    cluster: str, path: Optional[Path] = None, max_age_days: int = STATE_MAX_AGE_DAYS
) -> Optional[dict]:
    """Return the saved answers dict for `cluster` iff its per-cluster entry exists and is <max_age_days old;
    else None (and RMW-drops just that cluster's stale entry — never the whole map). Same 7-day +
    fromisoformat check as SLURM's load_install_state."""
    p = path or init_state_path()
    clusters = _load_state_map(p)
    entry = clusters.get(cluster)
    if not isinstance(entry, dict):
        return None
    ts = str(entry.get("timestamp") or "")
    try:
        if datetime.now() - datetime.fromisoformat(ts) > timedelta(days=max_age_days):
            clear_init_state(cluster, p)
            return None
    except Exception:
        clear_init_state(cluster, p)
        return None
    ans = entry.get("answers")
    return ans if isinstance(ans, dict) else None


def clear_init_state(cluster: Optional[str] = None, path: Optional[Path] = None) -> None:
    """Clear resume state. With `cluster` set → RMW-drop just that cluster's entry (a concurrent init for
    another cluster survives). With `cluster=None` → remove the whole file (test/reset helper).
    """
    p = Path(path or init_state_path())
    if cluster is None:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return
    clusters = _load_state_map(p)
    if cluster in clusters:
        clusters.pop(cluster, None)
        atomic_write_yaml(p, {"clusters": clusters})


# ─────────────────────────────────────────────────────────────────────────────
# Per-profile machine-readable readiness signal (gitignored, local-only)
#
# Written at the end of the Done-phase to cluster-profiles/.state/<profile>.readiness.json — atomic
# tmp+replace, mode 0600, same discipline as the profile write. Per-profile filename (= profile name) means
# concurrent inits for different clusters never collide. `.state/` is gitignored so it never enters the tree.
# Emit-only in Phase-1 — nothing reads it yet (a follow-on `fleet --grid` will). See wizard_init for the
# payload assembly (it reuses the STRUCTURED Check list + verdict()).
# ─────────────────────────────────────────────────────────────────────────────

READINESS_STATE_SCHEMA = 1


def readiness_state_path(profile: str, profiles_dir: Optional[Path] = None) -> Path:
    base = (
        Path(profiles_dir) if profiles_dir is not None else Path(__file__).resolve().parent.parent / "cluster-profiles"
    )
    return base / ".state" / f"{profile}.readiness.json"


def atomic_write_json(path: Path, data: dict) -> None:
    """Write `data` as JSON atomically + 0600 (same unique-tmp+O_EXCL+replace discipline as atomic_write_yaml)."""
    import json as _json

    _atomic_write(path, lambda f: _json.dump(data, f, indent=2, sort_keys=True))
