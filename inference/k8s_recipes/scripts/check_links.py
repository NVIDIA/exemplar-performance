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

"""Offline markdown link check — the local equivalent of the CI `Markdown Link Check` job.

CI runs (image lycheeverse/lychee:latest-alpine, from the REPO ROOT):

    lychee --offline --include-fragments -f detailed -q --no-progress \\
      --exclude-path "deprecated" --exclude-path "common" --exclude-path "scripts" \\
      "**/*.md"

That job checks the WHOLE repository, so one stale link anywhere blocks every MR. It also
only exists in CI, i.e. it can only tell you after you have pushed. This script is the same
gate, offline and dependency-free (stdlib only, python3.8+, no network, no docker), so
`make link-check` / `make ci` catches it before the push.

Scope, matching the CI invocation:
  * every `*.md` in the repo, minus any path containing a `deprecated` / `common` /
    `scripts` component;
  * LOCAL links only (`--offline`): http(s)/mailto/tel/etc are skipped, as in CI;
  * fragments (`--include-fragments`): `#anchor` is resolved against the target markdown
    file's headings (GitHub slugs) and explicit `id=` / `name=` anchors.

Exit 0 = clean, 1 = broken links (same contract as the CI job).
"""

import os
import re
import sys
import unicodedata
from urllib.parse import unquote

# --- CI parity: --exclude-path values from .gitlab/ci/stages/validate.yml -------------
EXCLUDED_PATH_PARTS = {"deprecated", "common", "scripts"}
# Never markdown source; excluded so we do not walk them.
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__"}

REMOTE_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
LOCAL_FILE_SCHEME = re.compile(r"^file:", re.I)

FENCE = re.compile(r"^(\s*)(```+|~~~+)")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)

# [text](dest "title")   — dest may be <bracketed>
INLINE_LINK = re.compile(
    r"\[(?:[^\[\]\\]|\\.|\[[^\[\]]*\])*\]\("
    r"\s*(<[^>\n]*>|[^\s()]*(?:\([^\s()]*\)[^\s()]*)*)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)"
)
# [label]: dest "title"
REF_DEF = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(<[^>\n]*>|\S+)")
HTML_ATTR = re.compile(r"<[a-zA-Z][^>]*?\b(?:href|src)\s*=\s*[\"']([^\"']+)[\"']")

ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
EXPLICIT_ANCHOR = re.compile(r"<[a-zA-Z][^>]*?\b(?:id|name)\s*=\s*[\"']([^\"']+)[\"']")


def strip_code(text):
    """Blank out fenced blocks and inline code spans, preserving line/column numbering.

    lychee parses markdown properly and does not treat `like/this` code spans as links;
    the repo has many backticked paths that are deliberately not links.
    """
    text = HTML_COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    out = []
    fence = None
    for line in text.split("\n"):
        m = FENCE.match(line)
        if fence is None:
            if m:
                fence = m.group(2)[0] * 3
                out.append(" " * len(line))
                continue
        else:
            out.append(" " * len(line))
            if m and m.group(2).startswith(fence):
                fence = None
            continue
        out.append(line)
    body = "\n".join(out)
    # Inline code spans (a run of backticks closed by an equal run). MUST be matched over the
    # whole document, not per line: a span may wrap across a newline, and matching per line
    # pairs the wrong backticks and silently blanks real links on the following line.
    return re.sub(
        r"(`+)(?:(?!\1)[\s\S])*?\1",
        lambda m: re.sub(r"[^\n]", " ", m.group(0)),
        body,
    )


def slug(heading_text):
    """GitHub heading -> anchor slug."""
    t = heading_text
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)  # images
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)  # links -> text
    t = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", t)  # ref links -> text
    t = t.replace("`", "")
    t = re.sub(r"[*_~]", "", t)  # emphasis / strike
    t = t.strip().lower()
    keep = []
    for ch in t:
        if ch in " -_":
            keep.append(ch)
        elif ch.isalnum() and unicodedata.category(ch)[0] in ("L", "N"):
            keep.append(ch)
        # everything else (punctuation, emoji, symbols) is dropped, as GitHub does
    # NOTE: no strip() here. GitHub drops the emoji but keeps the space it left behind, so
    # "## 🚀 Installation Steps" anchors as "#-installation-steps" (leading hyphen). Stripping
    # would report every emoji-heading link in the repo as a broken fragment.
    return "".join(keep).replace(" ", "-")


_anchor_cache = {}


def anchors_of(path):
    if path in _anchor_cache:
        return _anchor_cache[path]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
    except OSError:
        _anchor_cache[path] = None
        return None
    body = strip_code(raw)
    seen, anchors = {}, set()
    for line, orig in zip(body.split("\n"), raw.split("\n")):
        m = ATX_HEADING.match(line)
        if m:
            om = ATX_HEADING.match(orig)
            s = slug(om.group(2) if om else m.group(2))
            if not s:
                continue
            n = seen.get(s, 0)
            seen[s] = n + 1
            anchors.add(s if n == 0 else "%s-%d" % (s, n))
    for m in EXPLICIT_ANCHOR.finditer(raw):
        anchors.add(m.group(1))
    _anchor_cache[path] = anchors
    return anchors


def is_excluded(relpath):
    return any(part in EXCLUDED_PATH_PARTS for part in relpath.split(os.sep))


def markdown_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for fn in sorted(filenames):
            if not fn.endswith(".md"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            if is_excluded(rel):
                continue
            yield rel


def linkspecs(text):
    """Yield (offset, destination) for every link destination in the (code-stripped) text."""
    for m in INLINE_LINK.finditer(text):
        yield m.start(1), m.group(1)
    for m in HTML_ATTR.finditer(text):
        yield m.start(1), m.group(1)
    off = 0
    for line in text.split("\n"):
        m = REF_DEF.match(line)
        if m:
            yield off + m.start(1), m.group(1)
        off += len(line) + 1


def check_file(root, rel):
    """Return list of (line, col, dest, reason)."""
    full = os.path.join(root, rel)
    with open(full, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    body = strip_code(raw)
    line_starts = [0]
    for i, ch in enumerate(raw):
        if ch == "\n":
            line_starts.append(i + 1)

    def pos(off):
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1, off - line_starts[lo] + 1

    errors = []
    for off, dest in linkspecs(body):
        dest = dest.strip()
        if dest.startswith("<") and dest.endswith(">"):
            dest = dest[1:-1]
        if not dest:
            continue
        if REMOTE_SCHEME.match(dest) and not LOCAL_FILE_SCHEME.match(dest):
            continue  # --offline: remote links are not checked
        if dest.startswith("//"):
            continue
        if dest.startswith("{{") or "${" in dest:
            continue  # template placeholder, not a real link
        path_part, _, frag = dest.partition("#")
        path_part = unquote(path_part)
        frag = unquote(frag)
        if path_part.startswith("/"):
            # repo-absolute link
            target = os.path.join(root, path_part.lstrip("/"))
        elif path_part:
            target = os.path.normpath(os.path.join(os.path.dirname(full), path_part))
        else:
            target = full  # same-file fragment
        if not os.path.exists(target):
            errors.append(pos(off) + (dest, "File not found"))
            continue
        if frag and os.path.isfile(target) and target.endswith(".md"):
            anch = anchors_of(target)
            if anch is None:
                errors.append(pos(off) + (dest, "Target unreadable"))
            elif frag not in anch and frag.lower() not in anch:
                errors.append(pos(off) + (dest, "Cannot find fragment"))
    return errors


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    root = os.path.abspath(argv[1]) if len(argv) > 1 else default_root
    if not os.path.isdir(os.path.join(root, ".git")):
        # not a repo root (e.g. a worktree subdir) — still fine, just check what is here
        pass

    total = failed = 0
    by_file = []
    for rel in markdown_files(root):
        total += 1
        errs = check_file(root, rel)
        if errs:
            failed += len(errs)
            by_file.append((rel, errs))

    if by_file:
        print("Broken markdown links (offline check, same scope as CI 'Markdown Link Check'):\n")
        for rel, errs in by_file:
            print("Errors in %s" % rel)
            for line, col, dest, why in errs:
                print("  [ERROR] %s (at %d:%d) | %s" % (dest, line, col, why))
            print()
        print(
            "link-check FAILED — %d broken link(s) in %d file(s) (%d markdown files scanned)"
            % (failed, len(by_file), total)
        )
        print("This is the same gate CI runs; fixing it here keeps every other MR unblocked.")
        return 1

    print("link-check OK (%d markdown files, 0 broken local links/fragments)" % total)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
