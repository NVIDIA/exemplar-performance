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

"""Check that recipe and rendered Kubernetes resource names refer to the same cell.

This is an offline check used by CI, dry-run, and submit before any cluster resources are applied.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_NAME_RE = re.compile(r"^  name:[ \t]*(\S+)", re.M)  # envelope.name, 2-space indent
_META_RE = re.compile(r"^\s*name:[ \t]*[\"']?([A-Za-z0-9][A-Za-z0-9._-]*)", re.M)


def declared_name(cell: Path) -> str | None:
    """`envelope.name` from recipe.yaml, or None if unreadable/absent."""
    try:
        m = _NAME_RE.search((cell / "recipe.yaml").read_text())
    except OSError:
        return None
    return m.group(1).strip().strip("\"'") if m else None


def _metadata_names(text: str) -> list[str]:
    """Every `metadata.name` in a manifest. Only names directly under a `metadata:` block count —
    a bare `name:` elsewhere (env vars, ports, volumes) is not an object identity."""
    out, in_meta, meta_indent = [], False, 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("metadata:"):
            in_meta, meta_indent = True, indent
            continue
        if in_meta:
            if indent <= meta_indent:
                in_meta = False
            else:
                m = re.match(r"name:[ \t]*[\"']?([A-Za-z0-9][A-Za-z0-9._${}-]*)", stripped)
                if m and indent == meta_indent + 2:
                    out.append(m.group(1).strip("\"'"))
    return out


def manifest_names(cell: Path) -> dict[str, list[str]]:
    """{rendered-file-name: [metadata.name, ...]} for every rendered manifest."""
    found: dict[str, list[str]] = {}
    rendered = cell / "rendered"
    if not rendered.is_dir():
        return found
    for f in sorted(rendered.glob("*.yaml")):
        try:
            found[f.name] = _metadata_names(f.read_text())
        except OSError:
            found[f.name] = []
    return found


def check(cell: Path, *, other_cell_names: set[str] | None = None) -> list[str]:
    """Problems with this cell's identity. Empty list == consistent.

    Two distinct failures, both reported:
      * DRIFT   — a rendered object is not named for this cell (a partial rename)
      * FOREIGN — that name belongs to a DIFFERENT known cell, so applying it would collide with
                  somebody else's benchmark. This is the one that can corrupt a running experiment.
    """
    problems: list[str] = []
    name = declared_name(cell)
    if not name:
        return [f"{cell}: cannot read `envelope.name` from recipe.yaml — cell identity is UNKNOWN"]
    # One line per (file, offending name). A multi-doc manifest names several objects identically
    # (Job + ServiceAccount + Role all `<cell>-bench`), and repeating the same sentence once per
    # object buries the signal — the operator needs the distinct problems, not the object count.
    seen: set[tuple[str, str]] = set()
    for fname, names in manifest_names(cell).items():
        for got in names:
            if "${" in got:  # templated at apply time — resolved elsewhere, not a drift
                continue
            if got == name or got.startswith(f"{name}-"):
                continue
            if (fname, got) in seen:
                continue
            seen.add((fname, got))
            problems.append(
                f"{cell.name}: rendered/{fname} declares metadata.name={got!r} which is not "
                f"{name!r} nor {name}-* — a launcher builds resource names from recipe.yaml but "
                f"APPLIES this file, so it would create resources under the wrong identity"
                + (
                    f"  [FOREIGN: {got!r} belongs to another cell]"
                    if other_cell_names and _owner(got, other_cell_names, name)
                    else ""
                )
            )
    return problems


def _owner(obj_name: str, all_names: set[str], self_name: str) -> str | None:
    """The OTHER cell an object name belongs to, if any (longest match wins)."""
    best = None
    for n in all_names:
        if n == self_name:
            continue
        if obj_name == n or obj_name.startswith(f"{n}-"):
            if best is None or len(n) > len(best):
                best = n
    return best


def all_cell_names(recipes_root: Path) -> set[str]:
    names = set()
    for r in recipes_root.rglob("recipe.yaml"):
        n = declared_name(r.parent)
        if n:
            names.add(n)
    return names


def _main(argv: list[str]) -> int:
    """CLI: cell_identity.py <cell-dir> [<recipes-root>] — exit 0 consistent, 1 problems."""
    if not argv:
        print(__doc__)
        return 0
    cell = Path(argv[0])
    others = all_cell_names(Path(argv[1])) if len(argv) > 1 else None
    problems = check(cell, other_cell_names=others)
    for p in problems:
        print(f"  ✗ {p}", file=sys.stderr)
    if not problems:
        print(f"  ✓ cell identity consistent: {declared_name(cell)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
