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

"""Render a cell's serving-stack Jinja templates from its recipe.yaml.

Two substitution layers:
  {{ ... }}  Jinja  -> recipe truth, resolved here and baked into rendered/ (committed).
  ${ ... }   left AS-IS -> cluster truth, resolved at launch by envsubst from a cluster-profile.

Usage: _render.py <recipe.yaml> <templates_dir> <out_dir>
"""

import sys
from pathlib import Path

try:
    import yaml
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except ImportError:
    sys.exit("requires: pip install jinja2 pyyaml")


# Header added to every generated YAML file.
SPDX_HEADER = """\
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

"""


def main() -> int:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    recipe_path, templates_dir, out_dir = sys.argv[1:4]
    recipe = yaml.safe_load(Path(recipe_path).read_text()) or {}

    # Context = the recipe dict as-is, plus a few derived convenience helpers.
    ctx = dict(recipe)
    env_block = recipe.get("envelope") or {}
    prov = env_block.get("provenance") or {}
    ctx.setdefault("image_ref", prov.get("image_ref") or prov.get("image_digest") or "")
    ctx.setdefault("model", env_block.get("model", ""))

    tpl_dir = Path(templates_dir)
    templates = sorted(tpl_dir.glob("*.j2"))
    if not templates:
        sys.exit(f"no *.j2 templates in {tpl_dir}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jenv = Environment(
        # Stack dir first, then serving/_shared: a partial used by more than one stack (the
        # progress-liveness probe) lives in ONE place instead of being copied per stack, where
        # the copies would drift and only one would get fixed.
        loader=FileSystemLoader([str(tpl_dir), str(Path(tpl_dir).parent.parent / "_shared")]),
        undefined=StrictUndefined,  # fail loudly on a missing recipe field
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    written, skipped = [], []
    for t in templates:
        out = jenv.get_template(t.name).render(**ctx)
        dst = Path(out_dir) / t.stem  # drop the .j2 suffix
        # A fully-conditional template (the liveness ConfigMap on a cell that sets
        # liveness_enabled: false) renders to nothing. Writing a 0-byte manifest would leave an
        # artifact in the committed tree that LOOKS like a rendered object and applies as a silent
        # no-op — the absence-as-success shape. Emit no file, and delete a stale one so flipping the
        # flag off actually removes the object rather than leaving the last render behind.
        if not any(ln.strip() and not ln.lstrip().startswith("#") for ln in out.splitlines()):
            if dst.exists():
                dst.unlink()
            skipped.append(dst.name)
            continue
        dst.write_text(SPDX_HEADER + out)
        written.append(dst.name)
    print(
        f"rendered {len(written)} file(s) -> {out_dir}: {', '.join(written)}"
        + (f"  (skipped, rendered empty: {', '.join(skipped)})" if skipped else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
