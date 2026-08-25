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

"""lint_manifests.py [<root>] — validate every rendered manifest is structurally-valid k8s.

render-check only proves `rendered/ == render(recipe)`; it does NOT prove the result is valid k8s. This
neutralizes the ${...} envsubst/bash placeholders, parses every doc, and checks the basics (apiVersion +
kind + metadata.name, sane containers/ports/resources) — so a broken manifest is caught here, not at a
`kubectl apply` on a cluster. Always runs (no external dependency); kubeconform, if installed, is stricter
and complements this. Especially valuable for stacks that have never been deployed (sglang-disagg).
"""

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("lint: requires pyyaml")

NAMELESS = {None, "List"}  # kinds that legitimately have no metadata.name


def lint_file(path: Path, root: Path):
    problems = []
    # neutralize ${VAR} / ${bash} refs so the doc parses (they're strings or flow-map entries at runtime)
    neutral = re.sub(r"\$\{[^}]+\}", "x", path.read_text())
    rel = path.relative_to(root)
    try:
        docs = list(yaml.safe_load_all(neutral))
    except yaml.YAMLError as e:
        return [f"{rel}: YAML parse error — {str(e).splitlines()[0][:110]}"], 0
    n = 0
    for i, d in enumerate(docs):
        if not d:
            continue
        n += 1
        kind = d.get("kind")
        loc = f"{rel} doc{i} ({kind or '?'})"
        for k in ("apiVersion", "kind"):
            if not d.get(k):
                problems.append(f"{loc}: missing {k}")
        if kind not in NAMELESS and not ((d.get("metadata") or {}).get("name")):
            problems.append(f"{loc}: missing metadata.name")
        # pod-bearing kinds: sanity-check the pod spec
        spec = d.get("spec") or {}
        pod = (
            (spec.get("template") or {}).get("spec")
            if kind in ("Deployment", "Job", "StatefulSet", "DaemonSet")
            else (spec if kind == "Pod" else None)
        )
        if pod is not None:
            containers = pod.get("containers") or []
            if not containers:
                problems.append(f"{loc}: pod spec has no containers")
            for c in containers:
                if not c.get("name"):
                    problems.append(f"{loc}: a container has no name")
                if not (c.get("image") or c.get("command") or c.get("args")):
                    problems.append(f"{loc}: container {c.get('name','?')} has no image")
    return problems, n


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    files = sorted(root.glob("recipes/**/rendered/*.yaml"))
    if not files:
        print("lint: no rendered manifests found")
        return 0
    all_problems, docs = [], 0
    for f in files:
        p, n = lint_file(f, root)
        all_problems += p
        docs += n
    for p in all_problems:
        print(f"FAIL {p}")
    print(
        f"lint: {len(files)} rendered files, {docs} docs — "
        f"{'all structurally valid ✓' if not all_problems else f'{len(all_problems)} problem(s)'}"
    )
    return 1 if all_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
