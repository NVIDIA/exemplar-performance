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

"""selftest_runner_identity.py — WHO ran it (scripts/runner_identity.py).

Guards the fallback chain that stamps run_by on every export row — stored → git email → $USER → "unknown" —
and the user-level persistence (init writes it once; export reads it). No cluster, no real git, no real
~/.config: every case runs against a throwaway LLMB_CONFIG_DIR with git/$USER monkeypatched.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_identity as ri  # noqa: E402

_fail = 0


def check(label, cond):
    global _fail
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        _fail += 1


def _reset(cfg: Path, *, git_email: str, user: str):
    """Point config at a fresh temp dir; stub git email + $USER deterministically."""
    os.environ["LLMB_CONFIG_DIR"] = str(cfg)
    os.environ.pop("XDG_CONFIG_HOME", None)
    ri._git_email = lambda: git_email  # type: ignore[assignment]
    if user is None:
        os.environ.pop("USER", None)
    else:
        os.environ["USER"] = user


tmp = Path(tempfile.mkdtemp(prefix="selftest_runner_"))

# ── fallback chain: stored → git email → $USER → "unknown" ───────────────────────────────────────────────
_reset(tmp / "a", git_email="", user="")
check(
    "nothing anywhere → 'unknown' (never blocks/raises)",
    ri.resolve_runner() == "unknown",
)

_reset(tmp / "b", git_email="", user="testuser")
check("no stored, no git → falls back to $USER", ri.resolve_runner() == "testuser")

_reset(tmp / "c", git_email="testuser@example.com", user="testuser")
check(
    "no stored, git email present → git email wins over $USER",
    ri.resolve_runner() == "testuser@example.com",
)

# stored value wins over everything (identity was set at init and persists across clusters).
_reset(tmp / "d", git_email="testuser@example.com", user="testuser")
path = ri.save_runner("stored-user@example.com")
check(
    "save_runner persists user-level to <cfg>/user",
    path == (tmp / "d" / "user") and path.is_file(),
)
check(
    "stored identity wins over git email + $USER",
    ri.resolve_runner() == "stored-user@example.com",
)
check(
    "load_stored round-trips the saved value",
    ri.load_stored() == "stored-user@example.com",
)

# ── persistence details ──────────────────────────────────────────────────────────────────────────────────
_reset(tmp / "e", git_email="", user="testuser")
check(
    "save_runner('') is a no-op (never persist a blank identity)",
    ri.save_runner("  ") == (tmp / "e" / "user") and not (tmp / "e" / "user").is_file(),
)
check(
    "resolve with no stored blank still falls through to $USER",
    ri.resolve_runner() == "testuser",
)

_reset(tmp / "f", git_email="git@x", user="u")
ri.save_runner("first@example.com")
ri.save_runner("second@example.com")
check(
    "save_runner overwrites the run_by key idempotently",
    ri.load_stored() == "second@example.com",
)

# ── prompt_default: what init SHOWS as the bracket default (stored wins, else auto-detected) ─────────────
_reset(tmp / "g", git_email="detected@example.com", user="u")
check(
    "prompt_default offers the auto-detected default when nothing is stored",
    ri.prompt_default() == "detected@example.com",
)
ri.save_runner("chosen@example.com")
check(
    "prompt_default offers the stored identity once set (re-run confirms same person)",
    ri.prompt_default() == "chosen@example.com",
)

# ── config_dir override precedence (LLMB_CONFIG_DIR > XDG_CONFIG_HOME > ~/.config) ───────────────────────
os.environ["LLMB_CONFIG_DIR"] = "/tmp/explicit-llmb"
check(
    "LLMB_CONFIG_DIR overrides everything",
    ri.config_dir() == Path("/tmp/explicit-llmb"),
)
os.environ.pop("LLMB_CONFIG_DIR", None)
os.environ["XDG_CONFIG_HOME"] = "/tmp/xdg"
check(
    "XDG_CONFIG_HOME/llmb used when no explicit override",
    ri.config_dir() == Path("/tmp/xdg/llmb"),
)

print(f"\nselftest_runner_identity: {'all checks passed' if _fail == 0 else f'{_fail} FAILED'}")
raise SystemExit(1 if _fail else 0)
