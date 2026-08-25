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

"""Validate every recipes/**/recipe.yaml against schema/scenario-<scenario>.yaml.

The scenario schema embeds the shared envelope via a relative `$ref` to envelope.yaml, so we
resolve refs against the schema/ dir. Emits precise, field-level errors (the agent's feedback loop).
"""

import sys
import warnings
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

warnings.filterwarnings("ignore")  # RefResolver deprecation notice — keep the agent-facing output clean

try:
    import yaml
    from jsonschema import Draft202012Validator
    from jsonschema.validators import RefResolver  # deprecated but portable
except ImportError:
    sys.exit("requires: pip install jsonschema pyyaml")

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_DIR = ROOT / "schema"


def load(p):
    return yaml.safe_load(Path(p).read_text())


def main() -> int:
    recipes = sorted((ROOT / "recipes").glob("**/recipe.yaml"))
    if not recipes:
        print("validate: no recipe.yaml found yet (0 cells) — nothing to check")
        return 0

    base_uri = SCHEMA_DIR.as_uri() + "/"

    def yaml_handler(uri):  # resolve $ref -> load YAML (not JSON)
        parsed = urlparse(uri)
        return load(Path(url2pathname(parsed.path)))

    fails = 0
    for rp in recipes:
        rel = rp.parent.relative_to(ROOT)
        doc = load(rp)
        scenario = (doc.get("envelope") or {}).get("scenario")
        schema_path = SCHEMA_DIR / f"scenario-{scenario}.yaml"
        if not schema_path.exists():
            print(f"FAIL {rel}: envelope.scenario={scenario!r} has no schema/scenario-{scenario}.yaml")
            fails += 1
            continue
        schema = load(schema_path)
        resolver = RefResolver(base_uri=base_uri, referrer=schema, handlers={"file": yaml_handler})
        validator = Draft202012Validator(schema, resolver=resolver)
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        if errors:
            fails += 1
            print(f"FAIL {rel}:")
            for e in errors[:10]:
                loc = "/".join(map(str, e.path)) or "(root)"
                print(f"   - {loc}: {e.message}")
        else:
            print(f"OK   {rel}  [{scenario}]")

    passed = len(recipes) - fails
    print(f"validate: {passed}/{len(recipes)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
