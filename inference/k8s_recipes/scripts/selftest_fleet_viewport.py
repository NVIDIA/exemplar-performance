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

"""Verify that fleet watch output fits the terminal without silently hiding active state.

The tests cover explicit collapse markers, wrapped-line accounting, TTY viewport behavior, and clean
single-render output when stdout is not a terminal. They run offline with canned cluster responses.
"""

from __future__ import annotations

import os
import pathlib
import re
import select
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import fleet_render as FR  # the pure renderer under test
import selftest_fleet as SF  # fixture builders (kubectl shim, cluster profiles) — reused, not copied

SCRIPTS = pathlib.Path(__file__).resolve().parent
FLEET_SH = SCRIPTS / "fleet.sh"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


ANSI = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")


# ── a minimal VT screen, so "did it scroll off the top?" is MEASURED, not assumed ────────────────────────
# Subset the watch loop actually emits: \033[?1049h/l (alt screen), \033[?25l/h (cursor), \033[H (home),
# \033[J (clear-to-end), SGR, CR/LF. `scrolled` counts lines pushed off the TOP — in the alternate screen
# buffer those are UNRECOVERABLE, which is precisely the bug. Anything > 0 is a fail.
class _Screen:
    def __init__(self, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.g = [[" "] * cols for _ in range(rows)]
        self.y = self.x = 0
        self.scrolled = 0

    def nl(self):
        self.y += 1
        if self.y >= self.rows:
            self.g.pop(0)
            self.g.append([" "] * self.cols)
            self.y = self.rows - 1
            self.scrolled += 1

    def put(self, ch: str):
        if self.x >= self.cols:
            self.x = 0
            self.nl()
        self.g[self.y][self.x] = ch
        self.x += 1

    def text(self) -> str:
        return "\n".join("".join(r).rstrip() for r in self.g)


def emulate(data: str, rows: int, cols: int) -> _Screen:
    s = _Screen(rows, cols)
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        if c == "\033":
            m = re.match(r"\033\[([0-9;?]*)([a-zA-Z])", data[i:])
            if not m:
                i += 1
                continue
            p, f = m.group(1), m.group(2)
            if f == "H":
                nums = [int(x) if x else 1 for x in (p.split(";") if p else [])]
                s.y = (nums[0] if nums else 1) - 1
                s.x = (nums[1] if len(nums) > 1 else 1) - 1
                s.y = max(0, min(s.y, rows - 1))
                s.x = max(0, min(s.x, cols - 1))
            elif f == "J":
                k = int(p) if p.isdigit() else 0
                if k == 0:
                    for xx in range(s.x, cols):
                        s.g[s.y][xx] = " "
                    for yy in range(s.y + 1, rows):
                        s.g[yy] = [" "] * cols
                elif k == 2:
                    s.g = [[" "] * cols for _ in range(rows)]
            elif f == "K":
                for xx in range(s.x, cols):
                    s.g[s.y][xx] = " "
            elif f == "h" and p == "?1049":
                s = _Screen(rows, cols)  # alt screen starts empty and keeps NO scrollback
            i += m.end()
            continue
        if c == "\n":
            s.nl()
        elif c == "\r":
            s.x = 0
        elif c != "\x00":
            s.put(c)
        i += 1
    return s


# ── offline fixture env (canned kubectl + cluster profiles from selftest_fleet) ──────────────────────────
def _fixture_env(root: pathlib.Path) -> dict:
    fx = SF.write_fixtures(root)
    shim = SF.write_shim(root, fx)
    profiles = SF.write_profiles(root)
    env = dict(os.environ)
    env["PATH"] = f"{shim}{os.pathsep}{env['PATH']}"
    env["FLEET_PROFILES_DIR"] = str(profiles)
    env["FLEET_NOW"] = SF.NOW
    env["FLEET_FRAME_DEADLINE"] = "30"  # the cold first frame must complete under a fake-slow shim
    env["FLEET_WATCH_ITERATIONS"] = "1"
    env.pop("NO_COLOR", None)
    env["TERM"] = "xterm-256color"
    return env


def run_in_pty(env: dict, args: list, rows: int, cols: int, timeout: float = 120.0) -> str:
    """Run fleet.sh on a REAL pty of exactly (rows, cols) and return the raw byte stream it wrote."""
    import fcntl
    import pty
    import termios

    pid, fd = pty.fork()
    if pid == 0:  # child: exec, never return
        try:
            os.execvpe("bash", ["bash", str(FLEET_SH), *args], env)
        finally:
            os._exit(127)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    buf = b""
    while True:
        r, _, _ = select.select([fd], [], [], timeout)
        if not r:
            break
        try:
            d = os.read(fd, 65536)
        except OSError:
            break
        if not d:
            break
        buf += d
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass
    os.close(fd)
    return buf.decode("utf-8", "replace")


# ── 1. fit_viewport: the pure contract ───────────────────────────────────────────────────────────────────
def _unit_fit_viewport():
    print("\nfit_viewport (pure) — priority collapse + honest accounting")

    def frame(n_run=6, n_inst=8, n_cap=4, n_idle=5):
        lines, prios = [], []

        def add(txt, p, k=1):
            for i in range(k):
                lines.append(f"{txt} {i}")
                prios.append(p)

        add("HEADLINE", FR.P_RUN)
        add("cluster-bar", FR.P_STRUCT)
        add("capacity", FR.P_CAPACITY, n_cap)
        add("installed", FR.P_INSTALLED, n_inst)
        add("ns-bar", FR.P_STRUCT)
        add("RUN row", FR.P_RUN, n_run)
        add("", FR.P_BLANK, 3)
        add("idle cluster", FR.P_IDLE, n_idle)
        add("fleet build abc", FR.P_CHROME)
        return lines, prios

    lines, prios = frame()
    total = len(lines)

    # unbounded / already-fits → byte-identical passthrough (the one-shot pane must not change at all)
    check(
        "rows=0 (unbounded) returns the frame unchanged",
        FR.fit_viewport(lines, prios, 0) == lines,
    )
    check(
        "a frame that already fits is returned unchanged",
        FR.fit_viewport(lines, prios, total + 5) == lines,
    )

    # tall frame on a short viewport → fits, and says what it hid
    for rows in (10, 14, 20, 26):
        got = FR.fit_viewport(lines, prios, rows)
        check(
            f"fitted output never exceeds the {rows}-row budget",
            len(got) <= rows,
            f"got {len(got)} lines",
        )
        check(
            f"rows={rows}: an explicit viewport footer states the loss",
            any(FR.VP_FOOTER_MARK in g for g in got),
            got[-1] if got else "",
        )

    # THE INVARIANT: nothing hidden that is not counted. footer N == lines dropped, footer M == input size.
    # No footer is only legal when NOTHING was dropped (the frame fitted as-is).
    for rows in (8, 10, 14, 20, 26, 30, 40):
        got = FR.fit_viewport(lines, prios, rows)
        foot = [g for g in got if FR.VP_FOOTER_MARK in g]
        kept = [g for g in got if FR.VP_FOOTER_MARK not in g and FR.VP_MARK not in g]
        if not foot:
            ok, why = (
                len(kept) == total,
                "no footer but lines are missing — SILENT TRUNCATION",
            )
        else:
            m = re.search(r"(\d+) of (\d+) lines hidden", foot[0])
            ok = bool(m) and int(m.group(2)) == total and int(m.group(1)) == total - len(kept)
            why = foot[0]
        check(
            f"rows={rows}: accounting closes — footer 'N of M' == (dropped, original)",
            ok,
            f"{why} kept={len(kept)} total={total}",
        )

    # PRIORITY: the ladder eats the least-actionable end first and never reaches the run rows while
    # anything cheaper is still on the frame.
    got = FR.fit_viewport(lines, prios, 20)
    txt = "\n".join(got)
    check("live RUN rows survive a short viewport", txt.count("RUN row") == 6, txt)
    check("the headline survives a short viewport", "HEADLINE 0" in txt)
    check(
        "cluster/namespace structure survives (a run row with no cluster is ambiguous)",
        "cluster-bar 0" in txt and "ns-bar 0" in txt,
    )
    check("idle clusters are collapsed before run rows", "idle cluster 0" not in txt)
    check("the build stamp is collapsed before run rows", "fleet build abc 0" not in txt)
    check(
        "collapsed regions leave an in-place marker, not a hole",
        any(FR.VP_MARK in g and "hidden" in g for g in got),
        txt,
    )
    check(
        "the marker names WHAT was collapsed",
        "idle clusters" in txt or "installed inventory" in txt,
        txt,
    )

    # a viewport just barely short → only the cheapest thing goes (spacing/chrome), runs+installed intact
    got = FR.fit_viewport(lines, prios, total - 2)
    check(
        "a barely-short viewport keeps the installed inventory (cheapest regions go first)",
        "\n".join(got).count("installed") >= 8,
        "\n".join(got),
    )

    # LAST RESORT: even runs + structure overflow → tail cut, and the cut is stated
    lines2, prios2 = frame(n_run=60)
    got = FR.fit_viewport(lines2, prios2, 12)
    check(
        "overflow (runs alone exceed the viewport): still fits the budget",
        len(got) <= 12,
        str(len(got)),
    )
    foot = [g for g in got if FR.VP_FOOTER_MARK in g]
    check(
        "overflow is NAMED in the footer, not silent",
        bool(foot) and "overflow" in foot[0],
        foot[0] if foot else "no footer",
    )
    m = re.search(r"(\d+) of (\d+) lines hidden", foot[0]) if foot else None
    kept = [g for g in got if FR.VP_FOOTER_MARK not in g and FR.VP_MARK not in g]
    check(
        "overflow accounting still closes",
        bool(m) and int(m.group(2)) == len(lines2) and int(m.group(1)) == len(lines2) - len(kept),
        (foot[0] if foot else "") + f" kept={len(kept)}",
    )

    # WRAP: a long line costs more than one row, or "it fits" is a lie that scrolls the top away
    check(
        "a 200-char line costs 3 rows on an 80-col terminal",
        FR._vp_cost("x" * 200, 80) == 3,
    )
    check(
        "cols=0 means 'do not model wrap' (1 row per line)",
        FR._vp_cost("x" * 200, 0) == 1,
    )
    check(
        "ANSI color codes do not count toward the wrapped width",
        FR._vp_cost("\033[31m" + "x" * 40 + "\033[0m", 80) == 1,
    )
    for cols in (60, 80, 100, 160):
        got = FR.fit_viewport(lines, prios, 12, cols=cols)
        check(
            f"cols={cols}: the fitted frame (footer included) still fits 12 rows once WRAP is charged",
            sum(FR._vp_cost(g, cols) for g in got) <= 12,
            str(sum(FR._vp_cost(g, cols) for g in got)),
        )
    wide = (["W" * 240] * 10, [FR.P_RUN] * 10)
    got = FR.fit_viewport(wide[0], wide[1], 12, cols=80)
    check(
        "wrap is charged against the row budget (10×3-row lines do not 'fit' in 12 rows)",
        sum(FR._vp_cost(g, 80) for g in got) <= 12,
        str([FR._vp_cost(g, 80) for g in got]),
    )


# ── 2. the renderer honors --viewport-rows without changing the unbounded pane ───────────────────────────
def _unit_render_stages_viewport():
    print("\nrender_stages — viewport plumbed through, default pane untouched")
    with tempfile.TemporaryDirectory() as td:
        env = _fixture_env(pathlib.Path(td))
        env["NO_COLOR"] = "1"
        base = subprocess.run(["bash", str(FLEET_SH), "--stages"], capture_output=True, text=True, env=env)
        check(
            "one-shot --stages still renders (exit 0)",
            base.returncode == 0,
            base.stderr[:160],
        )
        n_base = len(base.stdout.splitlines())
        check(
            "one-shot --stages carries NO viewport footer (unbounded by default)",
            FR.VP_FOOTER_MARK not in base.stdout,
            str(n_base),
        )
        check(
            "the fixture frame is genuinely tall (the defect needs a tall frame)",
            n_base > 60,
            str(n_base),
        )


# ── 3. END-TO-END on a real pty: nothing may scroll off the alternate screen ─────────────────────────────
def _e2e_tty_heights():
    print("\nfleet.sh --watch on a TTY — the alt screen must never eat the top of the frame")
    with tempfile.TemporaryDirectory() as td:
        env = _fixture_env(pathlib.Path(td))
        for rows, cols in ((20, 200), (30, 200), (45, 160), (70, 200), (24, 100)):
            raw = run_in_pty(env, ["--watch", "1"], rows, cols)
            scr = emulate(raw, rows, cols)
            vis = ANSI.sub("", scr.text())
            check(
                f"{rows}x{cols}: NOTHING scrolls off the top of the alternate screen",
                scr.scrolled == 0,
                f"{scr.scrolled} lines lost",
            )
            check(
                f"{rows}x{cols}: the headline is on screen (top not eaten)",
                "ACTIVE  " in vis,
                vis[:80],
            )
            check(
                f"{rows}x{cols}: the live footer is on screen (bottom not eaten)",
                "↻ live" in vis,
                vis[-120:],
            )
            emitted = len(ANSI.sub("", raw).splitlines())
            if emitted > rows:  # the frame had to be trimmed → it must SAY so
                check(
                    f"{rows}x{cols}: a trimmed frame states the loss",
                    FR.VP_FOOTER_MARK in vis,
                    vis[-200:],
                )

        # A viewport tall enough for everything must NOT trim (no false 'lines hidden'). The height is
        # MEASURED from the unbounded frame rather than pinned to a literal: the pane grows (the MODEL
        # ledger added a section per namespace), and a hardcoded "tall enough" silently becomes "too short",
        # which fails as a false trim rather than as the real change it is.
        env_plain = dict(env)
        env_plain["NO_COLOR"] = "1"
        _one = subprocess.run(
            ["bash", str(FLEET_SH), "--stages"],
            capture_output=True,
            text=True,
            env=env_plain,
        )
        cols = 220
        rows = sum(FR._vp_cost(l, cols) for l in _one.stdout.splitlines()) + 8  # + the live footer & slack
        raw = run_in_pty(env, ["--watch", "1"], rows, cols)
        vis = ANSI.sub("", emulate(raw, rows, cols).text())
        check(
            "a tall-enough terminal renders the WHOLE frame with no viewport footer",
            FR.VP_FOOTER_MARK not in vis and "ACTIVE  " in vis,
            vis[-200:],
        )
        check(
            "a tall-enough terminal keeps every cluster section",
            all(f"CLUSTER {c}" in vis for c in ("alpha", "bravo", "charlie", "golf", "hotel")),
            "",
        )

        # the sections a campaign watcher is watching survive a SHORT terminal
        raw = run_in_pty(env, ["--watch", "1"], 70, 200)
        vis = ANSI.sub("", emulate(raw, 70, 200).text())
        check(
            "short terminal: live RUN / SERVER rows are what SURVIVES the collapse",
            vis.count("RUN / SERVER") >= 4 and "● RUNNING" in vis and "ORPHAN" in vis,
            vis[:200],
        )
        check(
            "short terminal: a ✗ FAILED run is not what gets dropped",
            "✗ FAILED" in vis,
            "",
        )
        check(
            "short terminal: the collapse is attributed in-place",
            FR.VP_MARK in vis and ("lines hidden" in vis or "line hidden" in vis),
            "",
        )


# ── 4. stdout is NOT a tty → one clean full render, never a truncatable stream ───────────────────────────
def _e2e_not_a_tty():
    print("\nfleet.sh --watch piped (stdout is not a TTY) — degrade to ONE full render")
    with tempfile.TemporaryDirectory() as td:
        env = _fixture_env(pathlib.Path(td))
        env.pop("FLEET_WATCH_ITERATIONS", None)  # unbounded: legacy this NEVER terminated
        env["NO_COLOR"] = "1"
        try:
            p = subprocess.run(
                ["bash", str(FLEET_SH), "--watch", "1"],
                capture_output=True,
                text=True,
                env=env,
                timeout=180,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            p = None
            timed_out = True  # PRE-FIX behaviour: the piped repaint loop simply never ends
        check(
            "piped --watch TERMINATES (a watch stream into a pipe can only be truncated)",
            not timed_out,
            "still running after 180s",
        )
        if timed_out:
            return
        check("piped --watch exits 0", p.returncode == 0, p.stderr[:200])
        check(
            "piped --watch emits NO alternate-screen escape",
            "\033[?1049h" not in p.stdout,
        )
        check("piped --watch emits NO cursor-home repaint", "\033[H" not in p.stdout)
        check(
            "piped --watch emits NO raw ANSI at all",
            "\033" not in p.stdout,
            repr(p.stdout[:60]),
        )
        check(
            "piped --watch emits exactly ONE frame (not a repeating stack)",
            p.stdout.count("ACTIVE  ") == 1,
            str(p.stdout.count("ACTIVE  ")),
        )
        check(
            "piped --watch renders the frame WHOLE (no viewport trimming without a terminal)",
            FR.VP_FOOTER_MARK not in p.stdout,
        )
        check(
            "piped --watch keeps the pane --watch asked for (--stages hierarchy)",
            "━━ CLUSTER alpha" in p.stdout,
            p.stdout[:120],
        )
        check(
            "piped --watch says on stderr WHY it is not watching",
            "not a TTY" in p.stderr or "needs a terminal" in p.stderr,
            p.stderr[:200],
        )
        check(
            "the degraded render is the COMPLETE frame (every cluster present)",
            all(f"CLUSTER {c}" in p.stdout for c in ("alpha", "bravo", "charlie", "golf", "hotel")),
            "",
        )

        # the bounded-frame test knob still forces the interactive path (existing selftest_fleet relies on it)
        env2 = dict(env)
        env2["FLEET_WATCH_ITERATIONS"] = "1"
        w = subprocess.run(
            ["bash", str(FLEET_SH), "--watch", "1"],
            capture_output=True,
            text=True,
            env=env2,
            timeout=180,
        )
        check(
            "FLEET_WATCH_ITERATIONS=N still forces the real watch path when piped (test escape hatch)",
            "\033[?1049h" in w.stdout and w.returncode == 0,
            repr(w.stdout[:20]),
        )


def main() -> int:
    _unit_fit_viewport()
    _unit_render_stages_viewport()
    _e2e_tty_heights()
    _e2e_not_a_tty()
    print()
    if fails:
        print(f"selftest_fleet_viewport: {len(fails)} FAILED")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("selftest_fleet_viewport: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
