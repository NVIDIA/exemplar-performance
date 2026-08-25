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

"""runner_identity.py — WHO ran a benchmark ("run_by"), the person-level attribution stamped on every
export row so a DB record can answer "who produced this number".

A runner is a PERSON, not a cluster property: one identity applies across every cluster an operator drives.
So it is stored USER-level (``~/.config/llmb/user``, XDG-aware) — NOT in the per-cluster profile .env — and
`llmb-k8s init` prompts for it once, persisting it so it is never re-asked. The export then reads it.

Fallback chain (resolve_runner) — NEVER raises, NEVER blocks an export on a missing runner:
    stored (~/.config/llmb/user)  →  git user.email  →  $USER  →  "unknown"

Pure/​IO split is unit-tested offline in selftest_runner_identity.py (point LLMB_CONFIG_DIR at a temp dir)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# The single stored key. The file is a tiny ``key=value`` env-style file (same shape as a cluster profile),
# so it stays greppable/hand-editable and future user-level settings can share it.
_KEY = "run_by"


def config_dir() -> Path:
    """The user-level llmb config dir. LLMB_CONFIG_DIR overrides (tests); else $XDG_CONFIG_HOME/llmb; else
    ~/.config/llmb. This is deliberately NOT the repo and NOT a cluster profile — identity is per-person."""
    override = os.environ.get("LLMB_CONFIG_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "llmb"


def config_file() -> Path:
    return config_dir() / "user"


def _read_kv(path: Path) -> dict:
    """Parse a tiny ``key=value`` file (blank lines / ``#`` comments ignored). Missing file → {}."""
    out: dict = {}
    if not path.is_file():
        return out
    try:
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    except OSError:
        return {}
    return out


def load_stored() -> str:
    """The persisted runner identity, or "" if unset. Reads the user-level config only."""
    return _read_kv(config_file()).get(_KEY, "").strip()


def save_runner(value: str) -> Path:
    """Persist the runner identity user-level (idempotent overwrite of the run_by key). Returns the file
    path. Creates the config dir. Empty/whitespace value is a no-op (we never persist a blank identity)."""
    value = (value or "").strip()
    path = config_file()
    if not value:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    kv = _read_kv(path)
    kv[_KEY] = value
    body = "".join(f"{k}={v}\n" for k, v in kv.items())
    path.write_text(body)
    return path


def _git_email() -> str:
    """git config user.email, or "" — never raises (git absent / not a repo / no email set are all fine)."""
    try:
        out = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (out.stdout or "").strip() if out.returncode == 0 else ""


def default_runner() -> str:
    """The AUTO-DETECTED default (before "unknown"): git user.email → $USER → "". This is what `init`
    offers as the prompt default and what resolve_runner falls back to when nothing is stored."""
    return _git_email() or os.environ.get("USER", "").strip() or ""


def prompt_default() -> str:
    """The value `llmb-k8s init` should SHOW as the bracket default: an already-stored identity wins (so a
    re-run confirms the same person), else the auto-detected default. May be "" (operator types one in)."""
    return load_stored() or default_runner()


def resolve_runner() -> str:
    """The runner stamped on an export row. Fallback chain, guaranteed non-empty, never raises:
    stored → git user.email → $USER → "unknown"."""
    return load_stored() or default_runner() or "unknown"


def main(argv=None) -> int:
    """CLI: `runner_identity.py` prints the resolved runner; `... --set <value>` persists one."""
    argv = list(argv if argv is not None else os.sys.argv[1:])
    if argv and argv[0] == "--set":
        val = argv[1] if len(argv) > 1 else ""
        p = save_runner(val)
        print(f"run_by={resolve_runner()}  (stored → {p})")
        return 0
    print(resolve_runner())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
