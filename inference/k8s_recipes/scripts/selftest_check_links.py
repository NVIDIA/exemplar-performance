#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

"""Regression tests for scripts/check_links.py (the offline `make link-check` gate).

A link checker that silently passes everything is worse than no link checker — it is
absence reported as zero. These cases pin the behaviours that were actually wrong on the
first cut and were only caught by diffing against lychee on a 256-file tree:

  * a code span that WRAPS A LINE must not swallow the next line's real link
    (per-line backtick pairing blanked a genuinely broken link and reported clean);
  * an emoji heading anchors with a LEADING HYPHEN ("## 🚀 Install" -> "#-install");
    stripping it flags every emoji-heading link in the repo as broken.
"""

import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "check_links.py")


def run(root):
    p = subprocess.run(
        [sys.executable, CHECKER, root], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True
    )
    return p.returncode, p.stdout


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


FAILURES = []


def expect(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def case(name, files, should_fail, must_mention=(), must_not_mention=()):
    root = tempfile.mkdtemp(prefix="linkcheck-")
    try:
        for rel, text in files.items():
            write(root, rel, text)
        rc, out = run(root)
        expect(name + " / exit", (rc != 0) == should_fail, "(rc=%d, wanted fail=%s)\n%s" % (rc, should_fail, out))
        for s in must_mention:
            expect(name + " / mentions %r" % s, s in out, "\n" + out)
        for s in must_not_mention:
            expect(name + " / silent about %r" % s, s not in out, "\n" + out)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    print("selftest_check_links:")

    case("dead file target", {"a.md": "see [x](missing.md)\n"}, True, must_mention=["missing.md", "File not found"])

    case("live file target", {"a.md": "see [x](b.md)\n", "b.md": "# B\n"}, False)

    case("link to a directory is NOT broken", {"a.md": "see [d](sub/)\n", "sub/b.md": "# B\n"}, False)

    case(
        "bad fragment",
        {"a.md": "see [x](b.md#nope)\n", "b.md": "# Getting Started\n"},
        True,
        must_mention=["Cannot find fragment"],
    )

    case("good fragment", {"a.md": "see [x](b.md#getting-started)\n", "b.md": "## Getting Started\n"}, False)

    # regression: emoji headings keep the space the emoji left behind
    case(
        "emoji heading anchors with a leading hyphen",
        {"a.md": "see [x](b.md#-installation-steps)\n", "b.md": "## \U0001f680 Installation Steps\n"},
        False,
    )

    # regression: a code span that wraps a line must not blank the next line's link
    case(
        "code span wrapping a newline does not hide the next line's broken link",
        {
            "a.md": "flow: `init -> install ->\nrun`; see [`b.md`](b.md#nope) and [`c.md`](c.md)\n",
            "b.md": "# B\n",
            "c.md": "# C\n",
        },
        True,
        must_mention=["Cannot find fragment"],
    )

    # a backticked path is prose, not a link
    case("backticked path is not a link", {"a.md": "the file `recipes/gone/recipe.yaml` was deleted\n"}, False)

    case(
        "fenced code block is not scanned",
        {"a.md": "```\n[x](missing.md)\n```\n"},
        False,
        must_not_mention=["missing.md"],
    )

    # remote links are never checked (--offline parity)
    case("remote links are skipped", {"a.md": "[x](https://example.invalid/nope) [y](mailto:a@b.c)\n"}, False)

    # CI parity: --exclude-path common/scripts
    case(
        "excluded paths are not scanned",
        {
            "common/b.md": "[x](missing.md)\n",
            "scripts/c.md": "[x](missing.md)\n",
        },
        False,
    )

    case(
        "reference-style definition is checked",
        {"a.md": "see [x][ref]\n\n[ref]: missing.md\n"},
        True,
        must_mention=["missing.md"],
    )

    case("html href is checked", {"a.md": '<a href="missing.md">x</a>\n'}, True, must_mention=["missing.md"])

    case(
        "explicit html anchor satisfies a fragment",
        {"a.md": "[x](b.md#custom)\n", "b.md": '<a id="custom"></a>\n# B\n'},
        False,
    )

    if FAILURES:
        print("\nselftest_check_links FAILED: %d case(s): %s" % (len(FAILURES), FAILURES))
        return 1
    print("selftest_check_links OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
