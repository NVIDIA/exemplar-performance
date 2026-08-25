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

"""run_id.py <cell> [--fit KIND] [--at STAMP] [--stamp <run_meta.json>] [--label LABEL] [--job-name KIND [--run-id ID]]

Mint a k8s-label-safe RUN id — the *instance* identity (one execution of an experiment), the analog of
training's `experiments/<desc>/<timestamp>` directory. The *experiment* (definition) is `recipe_hash` /
`experiment_id`; a run is one instance of it.

The id is DNS-1123 label-safe (≤63 chars, `[a-z0-9]([-a-z0-9]*[a-z0-9])?`). With `--fit <kind>` it is sized so
the whole Job name `<cell>-<kind>-<runid>` ALSO fits ≤63 — fixing the ">63-char label" failure without touching
the (name-baked) Job templates: for a long cell name the run-id shrinks but stays unique.

  run_id.py <cell> --fit bench          # what sweep.sh uses for RUN_ID   (job: <cell>-bench-<id>)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe_hash as _rh

try:
    import yaml
except ImportError:
    sys.exit("run_id: requires pyyaml")

MAX = 63
LABEL_RE = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
# An already-dated run-id: the DATE-LED auto formats mint()/date emit. A caller-supplied run-id that ALREADY
# matches one of these is a resume/re-run of an auto id — pass it through UNCHANGED (no double-prefix), so
# the resume path (which re-passes the dated id) stays deterministic. NOTE: mint() has a FOURTH auto format
# for long cell names — the tight base36-epoch token `t<b36>` — which is NOT date-led; it is handled by
# _is_auto_epoch_token() below (both are honored in mint_labeled). See #47.
DATED_RE = re.compile(r"^\d{6}t\d{4}-|^\d{8}t\d{6}|^\d{8}-\d{6}")


def sanitize(s: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", s.lower())).strip("-")


def _b36(n: int) -> str:
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    n, out = abs(n), ""
    while n:
        out = digits[n % 36] + out
        n //= 36
    return out or "0"


def _stamp(at: str | None) -> str:
    return at if at else datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S")  # 15 chars, DNS-safe


def _compact(stamp: str) -> str:
    """15-char UTC stamp → human, sortable YYMMDDtHHMM (11 chars). 20260722t031200 → 260722t0312."""
    return f"{stamp[2:8]}t{stamp[9:13]}"


def _epoch_b36(stamp: str) -> str:
    """base36 of the stamp's UTC epoch — short (~7 chars) and STILL sorts by time (same era ⇒ same width)."""
    dt = datetime.strptime(stamp, "%Y%m%dt%H%M%S").replace(tzinfo=timezone.utc)
    return _b36(int(dt.timestamp()))


# The OTHER auto format mint() can emit (besides the DATED_RE ones): the tight base36-epoch token
# `t<epoch_b36>` (e.g. 'ttiycpc'), used for LONG cell names where even the compact stamp won't fit the
# ≤63-char Job label. Like the dated forms, this is an AUTO-minted FINAL id — re-passing it as a --label
# must return it UNCHANGED. It was missing from DATED_RE, so an orchestrator (run.sh) that minted it and
# then handed it to the bench script (which re-labels the run-id) got a DIFFERENT id at bench time — the
# re-label re-stamps with a fresh now(), and bench-time != run.sh-start-time — so run.sh fetched
# /artifacts/<its id> while the driver wrote /artifacts/<bench-time id>: the #47 run-id mismatch. Matched
# structurally (leading 't', base36 body) AND semantically (decodes to a plausible UTC epoch) so an ordinary
# human label like 'toolbar' is never mistaken for one.
def _is_auto_epoch_token(label: str) -> bool:
    if len(label) < 2 or label[0] != "t":
        return False
    body = label[1:]
    if re.fullmatch(r"[0-9a-z]+", body) is None:
        return False
    try:
        dt = datetime.fromtimestamp(int(body, 36), tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return False
    return 2020 <= dt.year <= 2100


def mint(
    name: str,
    recipe_hash: str,
    at: str | None = None,
    fit_kind: str | None = None,
    maxlen: int = MAX,
) -> str:
    """Label-safe run-id, SORTABLE by time at every size. Preferred `YYMMDDtHHMM-<salt>` (e.g. 260722t0312-a1b2);
    with fit_kind, shrinks so <name>-<kind>-<id> ≤ maxlen — first the compact stamp, then a base36-epoch token
    `t<b36>` (still time-sortable, ~8 chars) for very long cell names. The salt (from stamp+recipe_hash)
    disambiguates same-minute runs; determinism (for --at / resume) is preserved."""
    stamp = _stamp(at)
    salt = _b36(int(hashlib.sha1(f"{stamp}{recipe_hash}".encode()).hexdigest(), 16))[-4:]
    full = f"{_compact(stamp)}-{salt}"  # 260722t0312-a1b2  (16 chars)
    if not fit_kind:
        return full
    budget = maxlen - len(sanitize(name)) - len(fit_kind) - 2  # the two '-' in <name>-<kind>-<id>
    if budget >= len(full):
        return full
    tight = f"t{_epoch_b36(stamp)}"  # t<b36 epoch>  (~8 chars, still sortable)
    if budget >= len(tight):
        return tight
    # For very long cell names, preserve recipe-derived entropy. The launch path atomically reserves the
    # identifier and adds a bounded counter if a collision remains.
    return salt[: max(budget, 1)]


def mint_labeled(
    name: str,
    label: str,
    at: str | None = None,
    fit_kind: str | None = None,
    maxlen: int = MAX,
) -> str:
    """Route a CALLER-SUPPLIED run-id label through the same stamp+fit machinery as an auto id, so a custom
    label lands in the path/Job-name/labels PREFIXED and time-sortable: `rp3` → `260728t0312-rp3`.

    - Idempotent: a label that ALREADY matches an auto/dated format (DATED_RE) is returned UNCHANGED — no
      double-prefix — so the resume path (which re-passes the dated id) stays deterministic.
    - Time-sortable: the compact stamp always leads.
    - Deterministic: `--at STAMP` fixes the prefix stamp (not now()).
    - Fits: with fit_kind, sized so <name>-<kind>-<id> ≤ maxlen — the compact stamp is kept (sortable) and
      the label slug is truncated; if even stamp+1-char won't fit (very long cell), fall back to the
      base36-epoch token `t<b36>` (still time-sortable), mirroring mint()."""
    label = (label or "").strip()
    if DATED_RE.match(label) or _is_auto_epoch_token(label):
        return label  # already an AUTO id — pass through, no re-prefix
        # (#47: the tight t<b36> form must be idempotent too,
        #  else run.sh's minted id ≠ the bench-time re-mint)
    stamp = _stamp(at)
    prefix = _compact(stamp)  # 260728t0312 (11 chars), leads → sortable
    slug = sanitize(label)
    full = f"{prefix}-{slug}" if slug else prefix
    if not fit_kind:
        return full
    budget = maxlen - len(sanitize(name)) - len(fit_kind) - 2  # the two '-' in <name>-<kind>-<id>
    if budget >= len(full):
        return full
    keep = budget - len(prefix) - 1  # room left for '-' + truncated slug
    if slug and keep >= 1:
        trimmed = slug[:keep].rstrip("-")
        if trimmed:
            return f"{prefix}-{trimmed}"  # keep leading stamp, shrink the label
    tight = f"t{_epoch_b36(stamp)}"  # brutally tight: sortable base36-epoch token
    if budget >= len(tight):
        return tight
    # Honor the actual budget (min 1) so the Job name stays ≤ maxlen for very long cell names — see mint().
    return tight[: max(budget, 1)]


def job_name(name: str, kind: str, run_id: str, maxlen: int = MAX) -> str:
    """The Job name <cell>-<kind>-<runid>, guaranteed ≤ maxlen by truncating the (redundant) name prefix."""
    tail = f"-{kind}-{run_id}"
    prefix = sanitize(name)[: maxlen - len(tail)].rstrip("-")
    return f"{prefix}{tail}" if prefix else run_id


def main() -> int:
    argv = sys.argv[1:]
    pos = [a for a in argv if not a.startswith("--")]
    if not pos:
        sys.exit(__doc__)
    cell = Path(pos[0]).resolve()
    rp = cell / "recipe.yaml"
    if not rp.is_file():
        sys.exit(f"run_id: no recipe.yaml at {cell}")
    name = ((yaml.safe_load(rp.read_text()) or {}).get("envelope") or {}).get("name") or cell.name
    rh = _rh.recipe_hash(cell)

    def opt(flag):
        return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None

    at = opt("--at")
    fit = opt("--fit")
    label = opt("--label")  # a CALLER-SUPPLIED run-id to stamp/fit (distinct from --job-name's --run-id)

    if "--job-name" in argv:
        kind = opt("--job-name")
        rid = opt("--run-id") or (
            mint_labeled(name, label, at, fit_kind=kind) if label is not None else mint(name, rh, at, fit_kind=kind)
        )
        print(job_name(name, kind, rid))
        return 0

    rid = mint_labeled(name, label, at, fit_kind=fit) if label is not None else mint(name, rh, at, fit_kind=fit)

    if "--stamp" in argv:
        meta_path = Path(opt("--stamp") or "")
        if not meta_path.is_file():
            sys.exit("run_id: --stamp needs an existing run_meta.json path")
        meta = json.loads(meta_path.read_text())
        meta["run_id"] = rid
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"stamped run_id={rid} into {meta_path}")
        return 0

    print(rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
