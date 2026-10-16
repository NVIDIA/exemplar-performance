#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Parse every recipe and config YAML in the repo, rejecting duplicate mapping keys.

PyYAML's safe_load silently keeps the last of two duplicate keys, which is the
most common hand-edit mistake in recipe metadata, so a stricter loader is used.
Run from the repo root; files come from `git ls-files`.
"""

import re
import subprocess
import sys

import yaml

PATTERN = re.compile(r"(^|/)(metadata|exemplar|release)\.yaml$")
EXTRA = {"cli/llmb-run/cluster_config.yaml", "cli/llmb-run/example_llmb_config.yaml"}


MERGE_TAG = "tag:yaml.org,2002:merge"


class StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            # `<<` has no constructor of its own; SafeLoader flattens it below.
            # Overriding a merged key is legitimate, so only literal keys count.
            if key_node.tag == MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
                seen.add(key)
            except TypeError:
                continue  # unhashable key; SafeLoader reports it itself
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
        return super().construct_mapping(node, deep=deep)


def main() -> int:
    files = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
    targets = sorted(f for f in files if f and (PATTERN.search(f) or f in EXTRA))
    if not targets:
        print("::error::no YAML files matched; check PATTERN/EXTRA in this script")
        return 1

    failed = 0
    for path in targets:
        try:
            with open(path, encoding="utf-8") as handle:
                yaml.load(handle, Loader=StrictLoader)
        except yaml.YAMLError as exc:
            failed += 1
            print(f"::error file={path}::{exc}")
    print(f"checked {len(targets)} files, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
