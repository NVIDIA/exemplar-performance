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

"""merge_imex_strip.py — deploy-time FLASHINFER strip: the graceful NVLink-IMEX degrade (C1 safety net).

Reads Kubernetes manifests from stdin and, when the cluster profile says the IMEX channel is NOT provisioned,
removes the forced `VLLM_ATTENTION_BACKEND=FLASHINFER` env from each GPU-serving Deployment container. Writes
the patched YAML to stdout. This is what turns "FLASHINFER-forcing cell resolved onto an IMEX-less cluster"
from a guaranteed CrashLoop on `cuMulticastCreate code=800` into a correct-but-slower run (vLLM auto-disables
only the allreduce_rms fusion — throughput only, zero quality impact).

This is the Tier-1 DEGRADE counterpart of the Tier-2 `merge_imex_claim.py` injection, and the apply-time home
of `capability_registry.relax_serving_env` (the pure strip decision). Exactly one of the two ever acts on a
given manifest, keyed on the profile state:

    NVLINK_MULTICAST_IMEX=provisioned    → merge_imex_claim injects the channel claim, FLASHINFER stays ON
    NVLINK_MULTICAST_IMEX=<anything else / unset>  → merge_imex_strip removes forced FLASHINFER (degrade)

AVAILABLE (CRD present but the pod claim un-wired) still fails code=800, so it is NOT safe-to-keep — only
PROVISIONED keeps FLASHINFER (capability_registry.SATISFIED_STATES). A manifest that never forced FLASHINFER
is passed through untouched (the natural recipe-scoping — nothing to strip).

Called from deploy.sh as a pipeline stage between envsubst and kubectl apply:

    envsubst "$wl" < server.yaml | merge_rdma_selector.py | merge_imex_claim.py | merge_imex_strip.py | kubectl apply -f -

Because it patches the LIVE apply stream (never the committed rendered/*.yaml), the recipe_hash of every cell
is unchanged — the strip is a cluster-truth apply-time concern, exactly like the claim injection and the RDMA
selector override (design §4.6). Exit 0 always — any failure falls back to passthrough so deploy is never
blocked (a slower-but-running server always beats a blocked deploy).
"""

import os
import sys

try:
    import yaml
except ImportError:
    sys.stdout.write(sys.stdin.read())
    sys.exit(0)

# Import the PURE strip builder from the registry (shared implementation for the decision).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from capability_registry import (
        FLASHINFER_ENV_NAME,
        SATISFIED_STATES,
        strip_forced_flashinfer,
    )
except Exception:  # registry unavailable → never block a deploy
    sys.stdout.write(sys.stdin.read())
    sys.exit(0)


def main() -> None:
    content = sys.stdin.read()
    state = (os.environ.get("NVLINK_MULTICAST_IMEX") or "").strip()

    # Only strip when the cluster does NOT have a provisioned channel to keep FLASHINFER alive. When
    # provisioned, merge_imex_claim wired the claim, so cuMulticastCreate succeeds — leave the env ON.
    if state in SATISFIED_STATES:
        sys.stdout.write(content)
        return

    try:
        docs = [d for d in yaml.safe_load_all(content) if d is not None]
        n_stripped = 0
        for doc in docs:
            _, n = strip_forced_flashinfer(doc, state)
            n_stripped += n
        if n_stripped:
            print(
                f"merge_imex_strip: NVLINK_MULTICAST_IMEX={state or '<unset>'} (not provisioned) — stripped "
                f"forced {FLASHINFER_ENV_NAME}=FLASHINFER from {n_stripped} GPU container(s); the server runs "
                "correct-but-slower (allreduce_rms fusion OFF) instead of CrashLooping on cuMulticastCreate "
                "code=800. Recover the throughput: llmb-k8s profile provision-imex --cluster <c>",
                file=sys.stderr,
            )
            sys.stdout.write(yaml.dump_all(docs, default_flow_style=False, allow_unicode=True))
        else:
            # Nothing to strip (no FLASHINFER-forcing GPU container) → byte-passthrough the ORIGINAL bytes,
            # symmetric to merge_imex_claim's no-op. Reserializing here would silently drop comments/flow-style
            # and the exact-bytes guarantee on the COMMON deploy path for no benefit.
            sys.stdout.write(content)
    except Exception as e:
        print(
            f"merge_imex_strip: warning — strip failed ({e}), applying unmodified",
            file=sys.stderr,
        )
        sys.stdout.write(content)


if __name__ == "__main__":
    main()
