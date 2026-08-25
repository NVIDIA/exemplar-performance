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

"""Verify that every resource created by this tooling has a documented reclaim path.

Creation sites are compared with a committed inventory. Each entry must name reclaim evidence or an
explicit unreclaimed reason. ``--update`` refreshes creator lists while leaving new entries unreviewed.
The check runs offline and does not modify cluster resources.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "scripts" / "testdata" / "resource_inventory.json"

# k8s kinds this collection can bring into existence. RBAC objects are grouped: they are
# namespace-scoped idempotent singletons and leak nothing measurable.
KINDS = [
    "Job",
    "CronJob",
    "Deployment",
    "Service",
    "Pod",
    "ConfigMap",
    "Secret",
    "PersistentVolumeClaim",
    "Lease",
    "DynamoGraphDeployment",
    "Namespace",
]
RBAC_KINDS = {
    "Role",
    "RoleBinding",
    "ServiceAccount",
    "ClusterRole",
    "ClusterRoleBinding",
}


def _scan_files() -> list[Path]:
    out: list[Path] = []
    for pat in (
        "scripts/*.sh",
        "scripts/*.py",
        "serving/**/*.j2",
        "serving/**/*.yaml",
        "serving/**/*.sh",
        "serving/**/*.py",
    ):
        out.extend(ROOT.glob(pat))
    return sorted(
        p
        for p in out
        if "selftest" not in p.name and "testdata" not in str(p) and "fixtures" not in str(p) and p.is_file()
    )


def _kind_patterns(kind: str) -> list[re.Pattern]:
    """A kind is 'created' when it appears as a manifest key — in a template, in a shell heredoc,
    in an `echo "kind: X"` line, or as a JSON dict built in python."""
    return [
        re.compile(rf'kind:\s*["\']?{kind}\b'),  # yaml / j2 / heredoc / echo "kind: X"
        re.compile(rf'"kind"\s*:\s*"{kind}"'),  # python/JSON dict
    ]


def discover() -> dict[str, list[str]]:
    found: dict[str, set[str]] = {}
    api_create = re.compile(r"create_namespaced_(\w+)")
    # A process that deliberately outlives its caller — the 72-orphan class.
    detach = re.compile(r"\bnohup\s|\bos\.setsid\(\)|\bdisown\b")
    for p in _scan_files():
        try:
            text = p.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        rel = str(p.relative_to(ROOT))
        for kind in KINDS:
            if any(rx.search(text) for rx in _kind_patterns(kind)):
                found.setdefault(kind, set()).add(rel)
        for kind in RBAC_KINDS:
            if any(rx.search(text) for rx in _kind_patterns(kind)):
                found.setdefault("RBAC", set()).add(rel)
        for m in api_create.finditer(text):
            found.setdefault(f"api:{m.group(1)}", set()).add(rel)
        if detach.search(text):
            found.setdefault("local:detached-process", set()).add(rel)
    return {k: sorted(v) for k, v in sorted(found.items())}


def load_fixture() -> dict:
    if not FIXTURE.exists():
        return {"resources": {}}
    return json.loads(FIXTURE.read_text())


def update() -> None:
    disc = discover()
    old = load_fixture().get("resources", {})
    res = {}
    for kind, creators in disc.items():
        prev = old.get(kind, {})
        entry = {"creators": creators, "reclaim": prev.get("reclaim", [])}
        if prev.get("unreclaimed"):
            entry["unreclaimed"] = prev["unreclaimed"]
        elif not entry["reclaim"]:
            entry["unreclaimed"] = (
                "UNREVIEWED: name the reclaim path, or state why this resource " "is deliberately never reclaimed."
            )
        res[kind] = entry
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(
        json.dumps(
            {
                "_doc": "See scripts/check_create_destroy.py. Every resource type needs `reclaim` "
                "[{file, evidence}] whose evidence string must still appear in that file, or an "
                "explicit `unreclaimed` reason. UNREVIEWED fails CI.",
                "resources": res,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"check_create_destroy: wrote {FIXTURE.relative_to(ROOT)} ({len(res)} resource types).")


def main(argv: list[str]) -> int:
    if "--update" in argv:
        update()
        return 0
    disc = discover()
    fx = load_fixture().get("resources", {})
    fails: list[str] = []

    for kind in sorted(set(disc) | set(fx)):
        if kind not in fx:
            fails.append(
                f"NEW resource type '{kind}' created by {disc[kind]} — nothing records what "
                f"reclaims it. Run `python3 scripts/check_create_destroy.py --update` and "
                f"write the reclaim path (or an explicit unreclaimed reason)."
            )
            continue
        if kind not in disc:
            fails.append(
                f"resource type '{kind}' is in the inventory but no creator was found — " f"rerun --update to prune it."
            )
            continue
        entry = fx[kind]
        new_creators = sorted(set(disc[kind]) - set(entry.get("creators", [])))
        gone = sorted(set(entry.get("creators", [])) - set(disc[kind]))
        if new_creators:
            fails.append(
                f"'{kind}': NEW creator(s) {new_creators} — confirm the recorded reclaim "
                f"path covers them, then rerun --update."
            )
        if gone:
            fails.append(f"'{kind}': recorded creator(s) {gone} no longer create it — rerun --update.")
        reclaim = entry.get("reclaim") or []
        unrec = entry.get("unreclaimed") or ""
        if not reclaim and not unrec:
            fails.append(f"'{kind}': neither a reclaim path nor an unreclaimed reason.")
        if unrec.startswith("UNREVIEWED"):
            fails.append(f"'{kind}': UNREVIEWED — a human must name the reclaim path or the reason.")
        for r in reclaim:
            f, ev = ROOT / r["file"], r["evidence"]
            if not f.is_file():
                fails.append(f"'{kind}': reclaim path file {r['file']} is missing.")
            elif ev not in f.read_text():
                fails.append(
                    f"'{kind}': the reclaim path in {r['file']} no longer contains "
                    f"{ev!r} — the destroy path was renamed or removed while the create "
                    f"path stayed."
                )

    unreclaimed = [k for k, v in fx.items() if not v.get("reclaim") and v.get("unreclaimed")]
    print(
        f"check_create_destroy: {len(disc)} resource type(s) created; "
        f"{len(disc) - len(unreclaimed)} with a named reclaim path; "
        f"{len(unreclaimed)} deliberately unreclaimed."
    )
    for k in unreclaimed:
        print(f"  UNRECLAIMED  {k}: {fx[k]['unreclaimed']}")
    if fails:
        print("\nFAIL — create without destroy:")
        for m in fails:
            print(f"  FAIL  {m}")
        return 1
    print("check_create_destroy OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
