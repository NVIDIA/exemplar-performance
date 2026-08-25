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

"""selftest_install_interactive.py — `llmb-k8s install` with NO arguments: pick a cluster, then recipes.

Offline. No cluster is touched: every kubectl call goes through an injected fake runner and every prompt
through an injected fake `input`.

What this pins down:
  1. ADDITIVE, not a swap — `install <cluster>` never prompts, whatever the profile list looks like.
  2. The prompt-loop class of bug we keep hitting: an advertised default that is not actually selectable,
     or a prompt that never terminates. Every prompt here must have an ACHIEVABLE default and must end —
     including on garbage input, on 'q', and on EOF (a piped stdin that runs dry).
  3. Not a tty → NEVER prompt. Fail with the argument to pass. (CI must not hang.)
  4. Skip the question when there's nothing to ask: exactly one profile, or $LLMB_CLUSTER names one.
  5. HONEST install state: install reports what fleet reports. A Bound-but-unvouched-for PVC must NOT read as
     ready, and an unreadable cluster must say NOTHING rather than "absent".
  6. Idempotency: `--recipes <already-installed>` is a no-op, not a re-download; `--reinstall` overrides.
"""

from __future__ import annotations

import io
import contextlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import install  # type: ignore[import]
import profile_resolver as pr  # type: ignore[import]

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def _profiles_dir(tmp: Path, specs: dict[str, dict]) -> Path:
    d = tmp / "cluster-profiles"
    d.mkdir(parents=True, exist_ok=True)
    for name, env in specs.items():
        (d / f"{name}.env").write_text("".join(f'{k}="{v}"\n' for k, v in env.items()))
    # decoys the lister must ignore
    (d / "_template.env").write_text("NAMESPACE=x\n")
    (d / "sample.env.example").write_text("NAMESPACE=x\n")
    return d


class Prompt:
    """A scripted `input` that RECORDS calls and refuses to be asked more than it was scripted for — an
    unbounded prompt loop shows up as EOFError here instead of hanging the test run."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls: list[str] = []

    def __call__(self, text=""):
        self.calls.append(text)
        if not self.answers:
            raise EOFError("prompt asked more times than scripted")
        return self.answers.pop(0)


def _krun_ctx(ctx: str):
    def _run(args, timeout=30):
        if args[:2] == ["config", "current-context"]:
            return 0, ctx + "\n", ""
        return 1, "", "not stubbed"

    return _run


_KRUN_DEAD = lambda args, timeout=30: (1, "", "Unable to connect to the server")  # noqa: E731


def main() -> int:
    print("selftest_install_interactive")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        d3 = _profiles_dir(
            tmp,
            {
                "alpha-b200": {"NAMESPACE": "ns-a", "GPU_PRODUCT": "NVIDIA-B200", "KUBE_CONTEXT": "ctx-a"},
                "bravo-gb300": {"NAMESPACE": "ns-b", "GPU_PRODUCT": "GB300", "KUBE_CONTEXT": "ctx-b"},
                "charlie-b200": {"NAMESPACE": "ns-c", "GPU_PRODUCT": "B200", "KUBE_CONTEXT": "ctx-c"},
            },
        )
        d1 = _profiles_dir(tmp / "one", {"solo-b200": {"NAMESPACE": "ns", "GPU_PRODUCT": "B200"}})
        three = pr.list_profiles(d3)
        check(
            "list_profiles ignores _template/.example",
            three == ["alpha-b200", "bravo-gb300", "charlie-b200"],
            str(three),
        )

        # ── 1. the existing contract: an explicit cluster NEVER prompts ─────────────────────────────
        p = Prompt([])
        got, err = install.resolve_cluster_arg(
            "bravo-gb300", is_tty=True, profiles=three, krun=_KRUN_DEAD, prompt=p, profiles_dir=d3
        )
        check("explicit <cluster> passes through untouched", (got, err) == ("bravo-gb300", ""), f"{got!r} {err!r}")
        check("explicit <cluster> asks NOTHING", p.calls == [], str(p.calls))
        # even a cluster that isn't a known profile is passed through — resolution/error stays where it was
        got, err = install.resolve_cluster_arg(
            "not-a-profile", is_tty=True, profiles=three, krun=_KRUN_DEAD, prompt=Prompt([]), profiles_dir=d3
        )
        check(
            "unknown explicit <cluster> still passes through (resolver owns that error)",
            got == "not-a-profile" and err == "",
        )

        # ── 2. nothing to ask ──────────────────────────────────────────────────────────────────────
        p = Prompt([])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            got, err = install.resolve_cluster_arg(
                None, is_tty=True, profiles=pr.list_profiles(d1), krun=_KRUN_DEAD, prompt=p, profiles_dir=d1
            )
        check(
            "exactly ONE profile → auto-selected, no prompt",
            (got, err, p.calls) == ("solo-b200", "", []),
            f"{got!r} {err!r} {p.calls}",
        )
        check("…and it says which one it used", "solo-b200" in out.getvalue())

        p = Prompt([])
        got, err = install.resolve_cluster_arg(
            None, is_tty=True, profiles=[], krun=_KRUN_DEAD, prompt=p, profiles_dir=d3
        )
        check(
            "ZERO profiles → actionable error pointing at init, no prompt",
            got == "" and "llmb-k8s init" in err and p.calls == [],
            f"{got!r} {err!r}",
        )

        # ── 3. not a tty → never prompt, name the argument ─────────────────────────────────────────
        p = Prompt([])
        got, err = install.resolve_cluster_arg(
            None, is_tty=False, profiles=three, krun=_KRUN_DEAD, prompt=p, profiles_dir=d3
        )
        check("non-tty → no prompt at all", p.calls == [], str(p.calls))
        check("non-tty → error names the argument to pass", "install <cluster-profile>" in err, err)
        check("non-tty → error lists the available profiles", all(n in err for n in three), err)
        check("non-tty → no cluster returned", got == "")

        # ── 4. the default must be REAL and ACHIEVABLE ─────────────────────────────────────────────
        dflt = install.default_profile(three, environ={}, current_ctx="", profiles_dir=d3)
        check("default with no signal = first profile (a real member)", dflt in three and dflt == three[0], dflt)
        dflt = install.default_profile(three, environ={}, current_ctx="ctx-c", profiles_dir=d3)
        check("default follows the CURRENT kube context", dflt == "charlie-b200", dflt)
        dflt = install.default_profile(
            three, environ={"LLMB_CLUSTER": "bravo-gb300"}, current_ctx="ctx-c", profiles_dir=d3
        )
        check("explicit $LLMB_CLUSTER beats the current context", dflt == "bravo-gb300", dflt)
        dflt = install.default_profile(three, environ={"LLMB_CLUSTER": "ghost"}, current_ctx="", profiles_dir=d3)
        check("a BOGUS $LLMB_CLUSTER never becomes an unselectable default", dflt in three, dflt)
        check(
            "no profiles → empty default (nothing to advertise)",
            install.default_profile([], environ={}, current_ctx="", profiles_dir=d3) == "",
        )

        menu = install.render_profile_menu(three, "charlie-b200", profiles_dir=d3)
        check("menu marks exactly ONE Enter-default", menu.count("(press Enter)") == 1, menu)
        check(
            "menu's default is a LISTED row",
            any("charlie-b200" in ln and "(press Enter)" in ln for ln in menu.splitlines()),
            menu,
        )
        check("menu numbers every profile 1..N", all(f"    {i}  " in menu for i in range(1, len(three) + 1)), menu)
        check("menu shows GPU + namespace so the pick is recognizable", "GB300" in menu and "ns=ns-b" in menu, menu)

        # ── 5. the picker terminates, always ───────────────────────────────────────────────────────
        p = Prompt([""])
        with contextlib.redirect_stdout(io.StringIO()):
            got = install.pick_profile_interactive(three, "bravo-gb300", prompt=p, profiles_dir=d3)
        check("Enter takes the advertised default", got == "bravo-gb300", got)
        check("Enter needed exactly one prompt", len(p.calls) == 1)
        check("the prompt TEXT advertises that same default", "[bravo-gb300]" in p.calls[0], p.calls[0])

        p = Prompt(["3"])
        with contextlib.redirect_stdout(io.StringIO()):
            got = install.pick_profile_interactive(three, "alpha-b200", prompt=p, profiles_dir=d3)
        check("a number picks that profile", got == "charlie-b200", got)

        p = Prompt(["bravo-gb300"])
        with contextlib.redirect_stdout(io.StringIO()):
            got = install.pick_profile_interactive(three, "alpha-b200", prompt=p, profiles_dir=d3)
        check("a profile NAME picks that profile", got == "bravo-gb300", got)

        p = Prompt(["9", "banana", "-1", ""])
        with contextlib.redirect_stdout(io.StringIO()):
            got = install.pick_profile_interactive(three, "alpha-b200", prompt=p, profiles_dir=d3)
        check(
            "garbage re-prompts then Enter still resolves (loop TERMINATES)",
            got == "alpha-b200" and len(p.calls) == 4,
            f"{got!r} {p.calls}",
        )

        p = Prompt(["q"])
        with contextlib.redirect_stdout(io.StringIO()):
            got = install.pick_profile_interactive(three, "alpha-b200", prompt=p, profiles_dir=d3)
        check("'q' quits (a distinct sentinel, not a silent pick)", got == install._PICK_QUIT, got)

        p = Prompt([])  # runs dry immediately → EOFError from the fake
        with contextlib.redirect_stdout(io.StringIO()):
            got = install.pick_profile_interactive(three, "alpha-b200", prompt=p, profiles_dir=d3)
        check("EOF (piped stdin ran dry) resolves to the default — never an infinite loop", got == "alpha-b200", got)

        with contextlib.redirect_stdout(io.StringIO()):
            got, err = install.resolve_cluster_arg(
                None, is_tty=True, profiles=three, krun=_krun_ctx("ctx-b"), prompt=Prompt([""]), profiles_dir=d3
            )
        check(
            "no-arg tty flow: Enter takes the current-context profile",
            (got, err) == ("bravo-gb300", ""),
            f"{got!r} {err!r}",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            got, err = install.resolve_cluster_arg(
                None, is_tty=True, profiles=three, krun=_krun_ctx("ctx-a"), prompt=Prompt(["q"]), profiles_dir=d3
            )
        check(
            "quitting the picker stops cleanly (no cluster, a message)",
            got == "" and "Cancel" in err,
            f"{got!r} {err!r}",
        )

    # ── 6. HONEST installed state (fleet's verdicts, not a kinder second opinion) ───────────────────
    prof = {"NAMESPACE": "ns", "MODEL_CACHE_PVC": "shared-cache", "GPU_PRODUCT": "B200"}
    cell = {
        "name": "c1",
        "gpu_type": "B200",
        "_path": "recipes/llm-perf/c1",
        "status": "wip",
        "scenario": "llm-perf",
        "goal": "pareto",
    }

    def _krun_json(pvcs, jobs, deploys):
        import json as _j

        def _run(args, timeout=30):
            kind = args[3] if len(args) > 3 else ""
            return 0, _j.dumps({"pvc": pvcs, "jobs": jobs, "deploy": deploys}.get(kind, {"items": []})), ""

        return _run

    def _pvc(name, phase="Bound", labels=None):
        return {
            "metadata": {"name": name, "namespace": "ns", "labels": labels or {}},
            "status": {"phase": phase},
            "spec": {"resources": {"requests": {"storage": "100Gi"}}},
        }

    claim = install.derive_recipe_cache(cell, prof)["name"]

    # a) Bound but nothing vouches for the contents → the empty-PVC trap. Must NOT read ready.
    live = install.probe_cluster_install_state(
        prof, krun=_krun_json({"items": [_pvc(claim)]}, {"items": []}, {"items": []})
    )
    glyph, desc = install.cell_cluster_state(cell, prof, live)
    check(
        "Bound-but-unverified cache is NOT reported as downloaded",
        "UNVERIFIED" in desc and glyph != install.GLYPH_READY,
        f"{glyph!r} {desc!r}",
    )

    # b) the PVC's own download-complete stamp → genuinely downloaded
    stamped = _pvc(claim, labels={"llmb.nvidia.com/download-complete": "true", "llmb.nvidia.com/model-name": "qwen3"})
    live = install.probe_cluster_install_state(
        prof, krun=_krun_json({"items": [stamped]}, {"items": []}, {"items": []})
    )
    glyph, desc = install.cell_cluster_state(cell, prof, live)
    check(
        "a stamped cache reads downloaded", "downloaded" in desc and glyph == install.GLYPH_READY, f"{glyph!r} {desc!r}"
    )

    # c) an UNREADABLE cluster must say nothing at all — RBAC denial is not an empty cluster
    live = install.probe_cluster_install_state(prof, krun=_KRUN_DEAD)
    check("unreadable cluster → readable=False", live["readable"] is False)
    check("unreadable cluster → cell state claims NOTHING", install.cell_cluster_state(cell, prof, live) == ("", ""))
    panel = install.render_installed_panel([cell], {}, prof, live, "some-cluster")
    check("panel says the cluster wasn't readable instead of implying 'absent'", "not readable" in panel, panel)

    # d) a PVC that never bound cannot serve weights
    live = install.probe_cluster_install_state(
        prof, krun=_krun_json({"items": [_pvc(claim, "Pending")]}, {"items": []}, {"items": []})
    )
    _, desc = install.cell_cluster_state(cell, prof, live)
    check("a Pending PVC is reported as a failure, not as installed", "Bound" in desc or "Pending" in desc, desc)

    # ── 7. idempotency of the headless door ────────────────────────────────────────────────────────
    installed_cell = dict(cell, recipe_hash="rh1")
    stamps = {
        "recipes/llm-perf/c1": {
            "cell": "recipes/llm-perf/c1",
            "recipe_hash": "rh1",
            "staged": {"stage": {"ok": True}},
            "preflight": "pass",
        }
    }
    check("precondition: the stamp marks it installed", install.cell_installed(installed_cell, stamps))
    with contextlib.redirect_stdout(io.StringIO()) as out:
        sel, errs = install.resolve_recipe_selection([installed_cell], stamps, "c1", False)
    check(
        "--recipes <already-installed> is a NO-OP (no re-stage, no re-download)",
        sel == [] and errs == [],
        f"{sel} {errs}",
    )
    check(
        "…and says so, with the escape hatch",
        "already installed" in out.getvalue() and "--reinstall" in out.getvalue(),
        out.getvalue(),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        sel, errs = install.resolve_recipe_selection([installed_cell], stamps, "c1", False, reinstall=True)
    check("--reinstall redoes it", len(sel) == 1 and errs == [], f"{sel} {errs}")
    with contextlib.redirect_stdout(io.StringIO()):
        sel, errs = install.resolve_recipe_selection([installed_cell], {}, "c1", False)
    check("a cell with NO stamp is still installed normally", len(sel) == 1)
    with contextlib.redirect_stdout(io.StringIO()):
        sel, errs = install.resolve_recipe_selection([installed_cell], stamps, "nope", False)
    check(
        "an unknown name is still an ERROR (not silently swallowed as 'installed')",
        sel == [] and len(errs) == 1,
        f"{sel} {errs}",
    )

    # ── 8. the recipe picker's own default is achievable and terminates ────────────────────────────
    cells = [dict(cell, name=f"c{i}", _path=f"recipes/llm-perf/c{i}", recipe_hash=f"rh{i}") for i in (1, 2)]
    groups, idx_map, gidx = install.build_recipe_menu(cells, {})
    chosen, err, _ = install.parse_recipe_selection("", idx_map, gidx)  # the advertised [all] default
    check(
        "recipe picker's Enter-default ('all') selects every selectable cell",
        err == "" and len(chosen) == len(idx_map) == 2,
        f"{err!r} {chosen}",
    )
    chosen, err, _ = install.parse_recipe_selection("none", idx_map, gidx)
    check("'none' is a real, non-destructive exit", err == "" and chosen == [])
    chosen, err, _ = install.parse_recipe_selection("99", idx_map, gidx)
    check(
        "an out-of-range number errors instead of picking something harmful",
        err != "" and chosen == [],
        f"{err!r} {chosen}",
    )
    # every group token the menu ADVERTISES must actually resolve — an advertised-but-dead token is the
    # same bug class as an unselectable default.
    for g in groups:
        _, err, _ = install.parse_recipe_selection(g["token"], idx_map, gidx)
        check(f"advertised group token {g['token']} resolves", err == "", err)
        for m in g["models"]:
            _, err, _ = install.parse_recipe_selection(m["token"], idx_map, gidx)
            check(f"advertised model token {m['token']} resolves", err == "", err)

    print()
    if fails:
        print(f"selftest_install_interactive: {len(fails)} FAILURE(S): {', '.join(fails)}")
        return 1
    print("selftest_install_interactive: all checks PASSED ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
