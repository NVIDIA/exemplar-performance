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

"""Check that shell examples can be pasted into bash or zsh.

Trailing comments are not safe in every interactive shell, so comments in fenced shell blocks must appear on their own line. Exit 1 when an unsafe line is found.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# A shell comment that FOLLOWS content on the same line. A whole-line comment is fine.
TRAILING = re.compile(r"^(?P<code>\s*[^\s#][^#]*?)\s+#\s*\S")
FENCE = re.compile(r"^\s*```\s*(\w+)?\s*$")
SKIP_DIRS = {".git", "node_modules"}


def hits(md: Path) -> list[tuple[int, str]]:
    out, in_bash = [], False
    for i, line in enumerate(md.read_text(errors="ignore").splitlines(), 1):
        m = FENCE.match(line)
        if m:
            in_bash = (m.group(1) or "").lower() in {"bash", "sh", "shell", "zsh"} if not in_bash else False
            continue
        if in_bash and TRAILING.match(line):
            out.append((i, line.rstrip()))
    return out


def main() -> int:
    problems = []
    for md in sorted(ROOT.glob("**/*.md")):
        if any(p in SKIP_DIRS for p in md.parts):
            continue
        for ln, text in hits(md):
            problems.append(
                f"{md.relative_to(ROOT)}:{ln}: trailing '#' comment in a bash block — "
                f"move it to its own line ABOVE the command (zsh pastes it as arguments)\n"
                f"    {text[:120]}"
            )
    if problems:
        print("NOT COPY-PASTEABLE — a trailing comment breaks these lines in zsh:\n")
        print("\n".join(problems))
        print(f"\ncheck_pasteable: {len(problems)} line(s) in bash blocks carry a trailing comment.")
        return 1
    print("check_pasteable OK (every bash block survives a literal paste into zsh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
