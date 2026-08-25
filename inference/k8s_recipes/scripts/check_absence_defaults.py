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

"""check_absence_defaults.py — fail CI when a FAILED read can render as a POSITIVE RESULT.

THE FAILURE CLASS (7+ instances in a single day). A read that did not land collapses into a value
that is byte-identical to a legitimate zero/empty, and that value then feeds a rendered claim or a
publication gate:

  * a failed `get pvc` rendered "— nothing installed —" over a namespace holding 721 GB of weights
  * `_split_llmb` returned ([], []) on a failed cluster-wide read — identical in shape to "we own no
    workloads" — which would have reclassified every artifacts PVC as leaked
  * a stats reader that missed its key printed n=0, so one of six required assertions never ran
  * missing / corrupt / deliberately-INVALID all collapsed into one None, then fell back to a
    survivor-inflated macro with no gate

THE RULE THIS ENFORCES.  **A read that did not land must be representable as UNKNOWN** — a value
DISTINCT from zero/empty. Concretely, inside the modules that render state or gate publication:

  Rule A (error-path collapse). A function that performs an external read (subprocess / kubectl /
      json.load / read_text / yaml.safe_load / open) must not answer its OWN error path with an
      empty container, 0, "" or False. Return None, raise, or return an explicit status sentinel —
      anything the caller cannot mistake for "I looked, and there was nothing."
  Rule B (inline read collapse). `<read(...)> or []` / `<read(...)>.get(k, 0)` in a function that
      has no error handling at all: the read's failure is erased at the point of use.
  Rule C (return code discarded). A `(rc, out, err)` read wrapper subscripted for its OUTPUT
      (`kubectl(...)[1]`) — the failure signal is thrown away at the call site, so an errored read
      and an empty one are the same string. Fixed shape, no false positives.
  Rule D (swallowed read). `except ...: pass` / `continue` wrapped around an external read: the
      variable the try body meant to set silently keeps its prior value, and a downstream
      "all N agree" style claim is then computed over N−k inputs without saying so.

WHY AN ALLOWLIST, NOT A BLANKET BAN. Blanket-flagging `or []` / `.get(k, 0)` produced 572 hits over
these same modules — pure noise nobody would read. Both rules above require an EXTERNAL READ in the
same function, which is what makes the signal ~7 findings instead of 572.

SCOPE is deliberately narrow: only modules that render cluster state or gate a published number.
Adding a module here is how you opt a new renderer/gate into the rule.

HOW IT FAILS AND HOW YOU CLEAR IT.
  * a finding NOT in the baseline  → CI FAILS. Fix it (model UNKNOWN), or record a verdict.
  * `--update` rewrites the baseline with verdict "UNREVIEWED"; UNREVIEWED entries FAIL, so the
    update path cannot be used to silence anything — a human must type a verdict.
  * verdict "known-defect" is allowed but RATCHETED: the count may never rise above the recorded
    ceiling, so the debt can only shrink.

Offline, no cluster, no network. Read-only.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "testdata" / "absence_defaults_baseline.json"

# Modules that RENDER cluster state or GATE a published number. Nothing else is in scope.
SCOPE = [
    "scripts/fleet_render.py",
    "scripts/fleet_status.py",
    "scripts/reclaim_storage.py",
    "scripts/reclaim.py",
    "scripts/repro_campaign_stats.py",
    "scripts/repro_campaign_analyze.py",
    "scripts/install.py",
    "scripts/publish.py",
]

# Calls that can FAIL for reasons unrelated to the answer being empty (cluster down, RBAC, truncated
# file, bad JSON). A function containing one of these is doing an external read.
READ_CALLS = {
    "run",
    "check_output",
    "check_call",
    "Popen",
    "communicate",
    "load",
    "loads",
    "safe_load",
    "read_text",
    "read_bytes",
    "read",
    "readlines",
    "open",
    "kubectl",
    "kc",
    "urlopen",
    "get_json",
}

# Wrappers that return (returncode, stdout, stderr). Subscripting one for [1]/[2] discards rc.
RC_TUPLE_WRAPPERS = {"kubectl", "krun", "default_krun", "kc", "run_kubectl", "_kubectl"}

VALID_VERDICTS = {
    # the caller genuinely CANNOT mistake this for a landed read (status enum / tri-state alongside)
    "unknown-modeled",
    # the value never becomes a POSITIVE claim or a gate — it renders as absence ("no value yet"),
    # is a display fallback, or reads a committed artifact whose parse another CI target enforces
    "cosmetic",
    # a real instance of the failure class, accepted as debt. RATCHETED — the count may only shrink.
    "known-defect",
}


# ── AST helpers ──────────────────────────────────────────────────────────────────────────────────
def _is_empty(node: ast.AST) -> bool:
    """True for the values a failed read must NEVER become: 0, "", False, [], {}, (), set()."""
    if isinstance(node, ast.Constant):
        return node.value is not None and node.value in (0, 0.0, "", False)
    if isinstance(node, ast.List) and not node.elts:
        return True
    if isinstance(node, ast.Dict) and not node.keys:
        return True
    if isinstance(node, ast.Tuple):
        # ("", "") and (None, []) both collapse: a tuple counts as empty when EVERY element is
        # empty-or-None and at least one element is a hard empty (a pure (None, None) models UNKNOWN).
        if not node.elts:
            return True
        elems = [(_is_empty(e), isinstance(e, ast.Constant) and e.value is None) for e in node.elts]
        return all(e or n for e, n in elems) and any(e for e, _ in elems)
    if isinstance(node, ast.Call):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        return name in ("set", "list", "dict", "tuple", "frozenset") and not node.args
    return False


def _own_nodes(fn: ast.AST):
    """Every node of `fn` EXCLUDING nested function bodies, so an inner helper is attributed to
    itself once rather than to every enclosing def."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(n))


def _is_broad_except(handler: ast.ExceptHandler) -> bool:
    """bare `except:` / `except Exception:` / `except BaseException:` — catches things the author
    never enumerated, which is what makes a silent `pass` here erase an unanticipated read failure.
    A NARROW handler (`except OSError: continue` in a per-file loop) is a deliberate skip, not this
    bug class, so it is not flagged."""
    t = handler.type
    if t is None:
        return True
    names = {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    return bool(names & {"Exception", "BaseException"})


def _read_call_in(node: ast.AST) -> str | None:
    for x in ast.walk(node):
        if isinstance(x, ast.Call):
            name = x.func.attr if isinstance(x.func, ast.Attribute) else getattr(x.func, "id", None)
            if name in READ_CALLS:
                return name
    return None


def _is_rc_failure_test(test: ast.AST) -> bool:
    """`if rc != 0:` / `if r.returncode:` / `if not ok:` — a read's own error branch."""
    src = ast.unparse(test)
    if not any(tok in src for tok in ("rc", "returncode", "ok", "exists", "is_file")):
        return False
    return any(tok in src for tok in ("!=", "not ", "== 0", "is None"))


def _snippet(node: ast.AST) -> str:
    return " ".join(ast.unparse(node).split())[:110]


def _fingerprint(module: str, func: str, rule: str, snippet: str) -> str:
    """Line-number-FREE identity, so an unrelated edit above a finding does not churn the baseline."""
    return f"{module}::{func}::{rule}::{snippet}"


# ── the two rules ────────────────────────────────────────────────────────────────────────────────
def scan_module(rel: str) -> list[dict]:
    path = ROOT / rel
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(), filename=rel)
    findings: list[dict] = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        own = list(_own_nodes(fn))
        read = next((r for r in (_read_call_in(n) for n in own) if r), None)
        if read is None:
            continue  # not an external read → out of scope, whatever it defaults
        has_guard = False
        for node in own:
            # Rule C — the (rc, out, err) return code is discarded at the call site.
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
                f = node.value.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                idx = node.slice.value if isinstance(node.slice, ast.Constant) else None
                if name in RC_TUPLE_WRAPPERS and idx in (1, 2):
                    findings.append(
                        dict(
                            module=rel,
                            func=fn.name,
                            rule="read-returncode-discarded",
                            line=node.lineno,
                            read=name,
                            snippet=_snippet(node),
                        )
                    )
            # Rule A1 — an except handler that answers with an empty value.
            if isinstance(node, ast.ExceptHandler):
                has_guard = True
                # Rule D — the handler swallows the read entirely.
                if _is_broad_except(node) and all(isinstance(s, (ast.Pass, ast.Continue)) for s in node.body):
                    findings.append(
                        dict(
                            module=rel,
                            func=fn.name,
                            rule="except-swallows-read",
                            line=node.lineno,
                            read=read,
                            snippet=f"except {_snippet(node.type) if node.type else ''}: "
                            f"{'pass' if isinstance(node.body[0], ast.Pass) else 'continue'}",
                        )
                    )
                for stmt in node.body:
                    if isinstance(stmt, ast.Return) and stmt.value is not None and _is_empty(stmt.value):
                        findings.append(
                            dict(
                                module=rel,
                                func=fn.name,
                                rule="except-returns-empty",
                                line=stmt.lineno,
                                read=read,
                                snippet=_snippet(stmt),
                            )
                        )
            # Rule A2 — the read's own rc/exists failure branch answers with an empty value.
            if isinstance(node, ast.If) and _is_rc_failure_test(node.test):
                has_guard = True
                for stmt in node.body:
                    if isinstance(stmt, ast.Return) and stmt.value is not None and _is_empty(stmt.value):
                        findings.append(
                            dict(
                                module=rel,
                                func=fn.name,
                                rule="read-failure-returns-empty",
                                line=stmt.lineno,
                                read=read,
                                snippet=_snippet(stmt),
                            )
                        )
        if has_guard:
            continue  # the function DOES branch on failure — Rule B would be noise
        # Rule B — no error handling anywhere, and the read's result is collapsed at the point of use.
        for node in own:
            if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or) and _is_empty(node.values[-1]):
                if any(_read_call_in(v) for v in node.values[:-1]):
                    findings.append(
                        dict(
                            module=rel,
                            func=fn.name,
                            rule="unguarded-read-or-empty",
                            line=node.lineno,
                            read=read,
                            snippet=_snippet(node),
                        )
                    )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) == 2
                and _is_empty(node.args[1])
                and _read_call_in(node.func.value)
            ):
                findings.append(
                    dict(
                        module=rel,
                        func=fn.name,
                        rule="unguarded-read-get-default",
                        line=node.lineno,
                        read=read,
                        snippet=_snippet(node),
                    )
                )
    return findings


def scan_all() -> list[dict]:
    out: list[dict] = []
    for rel in SCOPE:
        out.extend(scan_module(rel))
    # Fingerprints are line-free on purpose, so N identical shapes in one function dedupe to one
    # entry (one verdict covers them). `count` keeps the multiplicity visible in the report.
    dedup: dict[str, dict] = {}
    for f in out:
        fp = _fingerprint(f["module"], f["func"], f["rule"], f["snippet"])
        f["fingerprint"] = fp
        if fp in dedup:
            dedup[fp]["count"] += 1
        else:
            dedup[fp] = dict(f, count=1)
    return sorted(dedup.values(), key=lambda f: f["fingerprint"])


# ── baseline plumbing ────────────────────────────────────────────────────────────────────────────
def load_baseline() -> dict:
    if not BASELINE.exists():
        return {"known_defect_ceiling": 0, "entries": {}}
    return json.loads(BASELINE.read_text())


def write_baseline(findings: list[dict]) -> None:
    old = load_baseline()
    entries = {}
    for f in findings:
        prev = old["entries"].get(f["fingerprint"])
        entries[f["fingerprint"]] = (
            prev
            if prev
            else {
                "verdict": "UNREVIEWED",
                "why": "REQUIRED: state why a failed read here cannot be mistaken for a landed empty one.",
                "module": f["module"],
                "func": f["func"],
                "rule": f["rule"],
            }
        )
    ceiling = sum(1 for e in entries.values() if e.get("verdict") == "known-defect")
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    doc = old.get("_doc") or (
        "See scripts/check_absence_defaults.py. verdict must be one of "
        f"{sorted(VALID_VERDICTS)}; UNREVIEWED fails CI. 'known-defect' is RATCHETED by "
        "known_defect_ceiling - it may only shrink."
    )
    BASELINE.write_text(
        json.dumps(
            {
                "_doc": doc,
                "known_defect_ceiling": max(ceiling, old.get("known_defect_ceiling", 0)),
                "entries": dict(sorted(entries.items())),
            },
            indent=2,
        )
        + "\n"
    )


def main(argv: list[str]) -> int:
    findings = scan_all()
    if "--update" in argv:
        write_baseline(findings)
        print(
            f"check_absence_defaults: wrote {BASELINE.relative_to(ROOT)} "
            f"({len(findings)} findings). Any UNREVIEWED entry FAILS until a human gives a verdict."
        )
        return 0

    base = load_baseline()
    entries = base.get("entries", {})
    fails: list[str] = []
    new = [f for f in findings if f["fingerprint"] not in entries]
    for f in new:
        fails.append(
            f"NEW silent-absence default {f['module']}:{f['line']} in {f['func']}() " f"[{f['rule']}] — {f['snippet']}"
        )
    for f in findings:
        e = entries.get(f["fingerprint"])
        if e and e.get("verdict") not in VALID_VERDICTS:
            fails.append(
                f"UNREVIEWED verdict for {f['module']}:{f['line']} {f['func']}() "
                f"— a human must classify it {sorted(VALID_VERDICTS)}"
            )

    debts = [f for f in findings if entries.get(f["fingerprint"], {}).get("verdict") == "known-defect"]
    ceiling = base.get("known_defect_ceiling", 0)
    if len(debts) > ceiling:
        fails.append(f"known-defect RATCHET broken: {len(debts)} > ceiling {ceiling}")

    stale = [k for k in entries if k not in {f["fingerprint"] for f in findings}]

    print(
        f"check_absence_defaults: {len(findings)} finding(s) over {len(SCOPE)} in-scope modules; "
        f"{len(debts)} known-defect (ceiling {ceiling}); {len(stale)} stale baseline entr(ies)."
    )
    for f in debts:
        print(f"  DEBT  {f['module']}:{f['line']} {f['func']}() — " f"{entries[f['fingerprint']].get('why', '')}")
    for k in stale:
        print(f"  stale (fixed or moved; rerun --update to prune): {k[:100]}")
    if fails:
        print(
            "\nFAIL — a read that did not land must be representable as UNKNOWN, distinct from "
            "zero/empty. Return None / raise / return a status sentinel; if the value genuinely "
            "cannot become a claim, record a verdict via:\n"
            "    python3 scripts/check_absence_defaults.py --update   (then edit the verdict)\n"
        )
        for m in fails:
            print(f"  FAIL  {m}")
        return 1
    print("check_absence_defaults OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
