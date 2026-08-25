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

"""Print a guided command sequence for porting a recipe to another GPU or cluster target.

This command does not modify files. Operators can follow the printed validation and
rendering steps manually.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPU_ARCH = {
    "B200": "amd64",
    "GB200": "arm64",
    "GB300": "arm64",
}  # GB-class (Grace) = arm64


def arch_for(gpu: str) -> str:
    """PURE — kubernetes.io/arch for a GPU family. B200 = amd64; GB200/GB300 (Grace) = arm64."""
    return GPU_ARCH.get((gpu or "").upper(), "arm64")


def retarget_leaf(src_leaf: str, old_gpu: str, new_gpu: str) -> str:
    """PURE — swap the hardware token in a cell leaf slug (case-insensitive), keeping the rest.
    e.g. ('nemotron-ultra-3-gb200-vllm-agg', 'GB200', 'GB300') -> 'nemotron-ultra-3-gb300-vllm-agg'.
    """
    if not old_gpu:
        return src_leaf
    return re.sub(re.escape(old_gpu.lower()), new_gpu.lower(), src_leaf, flags=re.IGNORECASE)


def gb_flag_hint(new_gpu: str) -> str:
    """PURE — the hardware-class serving-flag note for the target (empty for B200)."""
    if arch_for(new_gpu) == "arm64":
        return (
            "add `--disable-custom-all-reduce` to serving.extra_args (the pinned vLLM custom-all-reduce "
            "path fails engine init on Grace/NVFP4); carry GB300-only tuning (FLASHINFER serving.env, "
            "startup_timeout_s) ONLY when targeting GB300"
        )
    return "B200 (amd64) — do NOT add the Grace `--disable-custom-all-reduce` / FLASHINFER flags"


def _read_gpu_type(cell: Path) -> str:
    rp = cell / "recipe.yaml"
    if not rp.is_file():
        return ""
    for ln in rp.read_text().splitlines():
        m = re.match(r"\s*gpu_type:\s*(\S+)", ln)
        if m:
            return m.group(1).strip()
    return ""


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    to_gpu = _opt("--to")
    cluster = _opt("--cluster")
    src = args[0] if args else "<source-cell>"
    src_cell = Path(src)
    old_gpu = _read_gpu_type(ROOT / src) if args else ""
    newcell = "recipes/<scenario>/<distribution>/<model>-<hardware>-<setup>"
    if args and to_gpu and old_gpu:
        newcell = str(src_cell.parent / retarget_leaf(src_cell.name, old_gpu, to_gpu))
    ncl = cluster or "<newcluster>"

    print(f"── port a recipe → new GPU/cluster target {'─' * 30}")
    print(f"  source: {src}" + (f"   ({old_gpu} → {to_gpu})" if (old_gpu and to_gpu) else ""))
    print("  command reference: docs/CLI.md\n")

    print("1. Clone the closest cell + drop stale outputs:")
    print(f"     cp -r {src} {newcell}")
    print(f"     rm -rf {newcell}/rendered {newcell}/record.json   # regenerate these")
    print()
    print("2. Retarget the ENVELOPE (hardware fields only — leave ${VARS}/node-group tolerations to the profile):")
    print(f"     $EDITOR {newcell}/recipe.yaml")
    print("     • envelope.name (swap the hardware token, keep the distribution suffix)")
    if to_gpu:
        print(
            f"     • envelope.gpu_type: {to_gpu.upper()}   • envelope.arch: {arch_for(to_gpu)}   "
            f"• pick the arch-correct image_digest"
        )
        print(f"     • serving flags: {gb_flag_hint(to_gpu)}")
    else:
        print("     • envelope.gpu_type: B200|GB200|GB300   • envelope.arch: B200=amd64, GB200/GB300=arm64")
        print("     • arch-correct image_digest; GB-class needs --disable-custom-all-reduce (see a GB300 cell)")
    print("     • requires.gpu.count if TP changes (e.g. B200 tp=8 → GB tp=4)")
    print("     • scenario-specific: fix any scenario-specific envelope fields")
    print()
    print("3. Validate the target cluster profile (or create it):")
    print(f"     scripts/llmb-k8s profile validate --cluster {ncl}")
    print(f"     # missing: cp cluster-profiles/_template.env.example cluster-profiles/{ncl}.env  &&  fill it")
    print()
    print("4. Render + the static target-compat gate (catches GB200-recipe-on-GB300 that the live guard misses):")
    print(f"     scripts/render.sh {newcell}")
    print(f"     python3 scripts/profile_resolver.py compat {ncl} {newcell}")
    print()
    print("5. CI, then commit recipe.yaml + its regenerated rendered/:")
    print("     make ci        # validate → contract → invariants → provenance → render-check → lint → test → matrix")
    print()
    print("6. Live-check against the target before a real run:")
    print(f"     scripts/llmb-k8s preflight --recipe {newcell} --cluster {ncl}")
    return 0


def _opt(flag: str):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else None
    return None


if __name__ == "__main__":
    raise SystemExit(main())
