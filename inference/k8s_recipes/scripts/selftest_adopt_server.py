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

"""selftest_adopt_server.py — offline guards for scripts/adopt_server.sh (the standard-lane pure-k8s GC
ownerReference wiring). No cluster: a fake `kubectl` shim (KUBECTL override) serves a canned Deployment
list and records every `patch` so we can assert exactly which servers get adopted and with what owner.

Covers the orphan-prevention contract:
  - refuses to stamp an ownerReference when the owner UID is empty (a wrong/empty uid would make GC delete
    the server IMMEDIATELY — leaving it un-owned is strictly safer),
  - adopts ONLY the cell's own server Deployment(s) (managed-by label + EXACT <cell>-server / <cell>-prefill /
    <cell>-decode name), never a sibling cell's server — including a sibling whose name STARTS WITH this cell's
    (…-1m vs …-1m-offload), which an open "^<cell>-" prefix would wrongly adopt + clobber,
  - the patched ownerReference is a non-controller, non-blocking Job reference with the real UID,
  - --dry-run mutates nothing (issues no patch).
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ADOPT = ROOT / "scripts" / "adopt_server.sh"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── 0. static lint ────────────────────────────────────────────────────────────
shn = subprocess.run(["bash", "-n", str(ADOPT)], capture_output=True, text=True)
check("adopt_server.sh is valid bash (bash -n)", shn.returncode == 0, shn.stderr.strip()[:200])


def _shim(tmp: Path, deploy_names: list[str]) -> tuple[Path, Path]:
    """Write a fake kubectl: `get deploy ... -o jsonpath` prints deploy_names; `patch` logs its args."""
    patch_log = tmp / "patch.log"
    listing = "".join(n + "\n" for n in deploy_names)
    shim = tmp / "kubectl"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        f'PATCH_LOG="{patch_log}"\n'
        'args="$*"\n'
        'case "$args" in\n'
        '  *"get deploy"*)  printf %s ' + repr_shell(listing) + '; exit 0 ;;\n'
        '  *"patch deploy"*) echo "$args" >> "$PATCH_LOG"; exit 0 ;;\n'
        '  *) exit 0 ;;\n'
        'esac\n'
    )
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return shim, patch_log


def repr_shell(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def run_adopt(tmp: Path, deploy_names: list[str], *args: str) -> subprocess.CompletedProcess:
    shim, _ = _shim(tmp, deploy_names)
    env = dict(os.environ)
    env["KUBECTL"] = str(shim)
    env.pop("KUBE_CONTEXT", None)
    return subprocess.run(["bash", str(ADOPT), *args], capture_output=True, text=True, env=env)


# ── 1. empty UID → refuse, no mutation ────────────────────────────────────────
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    p = run_adopt(tmp, ["cell-server"], "ns", "cell", "cell-bench-r1", "")  # empty uid
    _, patch_log = _shim(tmp, ["cell-server"])
    check(
        "empty uid: exits 0 and refuses to stamp",
        p.returncode == 0 and "empty owner uid" in p.stdout + p.stderr,
        p.stdout + p.stderr,
    )

# ── 2. --dry-run adopts only cell-prefixed servers; issues NO patch ───────────
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    p = run_adopt(
        tmp, ["cell-server", "cell-prefill", "other-server"], "ns", "cell", "cell-bench-r1", "uid-123", "--dry-run"
    )
    _, patch_log = _shim(tmp, [])  # recompute the (unused) log path deterministically
    dry = p.stdout
    check("dry-run: adopts cell-server", "deploy/cell-server -> job/cell-bench-r1" in dry, dry)
    check("dry-run: adopts cell-prefill", "deploy/cell-prefill -> job/cell-bench-r1" in dry, dry)
    check("dry-run: does NOT adopt sibling cell 'other-server'", "other-server" not in dry, dry)
    check("dry-run: issues no patch", not patch_log.exists(), "patch.log should not exist")

# ── 3. live: patches cell servers with a non-controller Job ownerRef carrying the real uid ────────────
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim, patch_log = _shim(tmp, ["cell-server", "cell-prefill", "other-server"])
    env = dict(os.environ)
    env["KUBECTL"] = str(shim)
    env.pop("KUBE_CONTEXT", None)
    p = subprocess.run(
        ["bash", str(ADOPT), "ns", "cell", "cell-bench-r1", "uid-123"], capture_output=True, text=True, env=env
    )
    logged = patch_log.read_text() if patch_log.exists() else ""
    check("live: patched cell-server", "patch deploy cell-server" in logged, logged)
    check("live: patched cell-prefill", "patch deploy cell-prefill" in logged, logged)
    check("live: did NOT patch sibling 'other-server'", "patch deploy other-server" not in logged, logged)
    check(
        "live: ownerReference carries the real uid + controller:false + blockOwnerDeletion:false",
        '"uid":"uid-123"' in logged and '"controller":false' in logged and '"blockOwnerDeletion":false' in logged,
        logged,
    )
    check("live: exits 0", p.returncode == 0, p.stdout + p.stderr)

# ── 4. no matching server yet → clean skip, no patch ──────────────────────────
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim, patch_log = _shim(tmp, ["other-server"])
    env = dict(os.environ)
    env["KUBECTL"] = str(shim)
    env.pop("KUBE_CONTEXT", None)
    p = subprocess.run(
        ["bash", str(ADOPT), "ns", "cell", "cell-bench-r1", "uid-123"], capture_output=True, text=True, env=env
    )
    check("no matching server: clean skip, exit 0", p.returncode == 0 and "nothing to adopt" in p.stdout, p.stdout)
    check("no matching server: no patch issued", not patch_log.exists(), "patch.log should not exist")

# ── 5. DESTRUCTIVE-SAFETY: a cell that is a strict PREFIX of a sibling never adopts the sibling's server ──
# `…-1m` must select ONLY …-1m-server / …-1m-decode, NEVER the sibling …-1m-offload-server (an open
# "^${NAME}-" prefix would match it, then `patch --type merge` clobbers the sibling's ownerReference so GC
# cascade-deletes the sibling's LIVE server when THIS cell's Job ends). Exact server-suffix match, anchored.
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    shim, patch_log = _shim(tmp, ["foo-1m-server", "foo-1m-decode", "foo-1m-offload-server", "foo-1m-server-extra"])
    env = dict(os.environ)
    env["KUBECTL"] = str(shim)
    env.pop("KUBE_CONTEXT", None)
    p = subprocess.run(
        ["bash", str(ADOPT), "ns", "foo-1m", "foo-1m-bench-r1", "uid-9"], capture_output=True, text=True, env=env
    )
    logged = patch_log.read_text() if patch_log.exists() else ""
    check("prefix-collision: adopts own foo-1m-server", "patch deploy foo-1m-server --type" in logged, logged)
    check("prefix-collision: adopts own foo-1m-decode", "patch deploy foo-1m-decode --type" in logged, logged)
    check(
        "prefix-collision: NEVER adopts sibling foo-1m-offload-server",
        "patch deploy foo-1m-offload-server" not in logged,
        logged,
    )
    check(
        "prefix-collision: NEVER adopts non-exact foo-1m-server-extra",
        "patch deploy foo-1m-server-extra" not in logged,
        logged,
    )
    check("prefix-collision: exits 0", p.returncode == 0, p.stdout + p.stderr)


print()
if fails:
    print(f"selftest_adopt_server: {len(fails)} FAILED: {fails}")
    sys.exit(1)
print("selftest_adopt_server: all checks passed")
