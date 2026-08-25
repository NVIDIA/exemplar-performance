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

"""merge_imex_claim.py — deploy-time NVLink-IMEX ComputeDomain channel-claim injection (Tier 2).

Reads Kubernetes manifests from stdin and, when the cluster profile says the IMEX channel is provisioned,
wires each GPU-serving Deployment to CLAIM a ComputeDomain channel (a DRA ResourceClaim) so the FlashInfer
`allreduce_rms` NVLink multicast fusion can build its FABRIC multicast object (cuMulticastCreate) instead of
failing with code=800. Writes the patched YAML to stdout.

This is the Tier-2 counterpart of the Tier-1 strip (scripts/merge_imex_strip.py): that strip removes forced
FLASHINFER to avoid the crash when the channel is un-provisioned; this injection ADDS the channel claim so
FLASHINFER can stay ON and the fusion runs — recovering throughput. Exactly one of the two acts on a given
manifest, keyed on NVLINK_MULTICAST_IMEX (provisioned → claim; anything else → strip).

Gated by the profile (cluster truth), NOT the recipe:
    NVLINK_MULTICAST_IMEX=provisioned    # set by `llmb-k8s profile provision-imex` after the CD is applied
    IMEX_CLAIM_TEMPLATE=llmb-imex-channel  # the ResourceClaimTemplate the ComputeDomain controller generates
When either is unset/other, passes stdin through unchanged (no-op) — so a cluster without a provisioned
ComputeDomain, or a run that didn't provision, renders exactly as before.

Called from deploy.sh as a pipeline stage between envsubst and kubectl apply:

    envsubst "$wl" < server.yaml | merge_rdma_selector.py | merge_imex_claim.py | kubectl apply -f -

Because it patches the LIVE apply stream (never the committed rendered/*.yaml), the recipe_hash of every cell
is unchanged — the ComputeDomain claim is a cluster-truth apply-time concern, exactly like the RDMA selector
override (design §4.6). Exit 0 always — any failure falls back to passthrough so deploy is never blocked.
"""

import os
import sys

try:
    import yaml
except ImportError:
    sys.stdout.write(sys.stdin.read())
    sys.exit(0)

# Import the PURE injection builder from the registry (shared implementation for the claim shape).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from capability_registry import IMEX_POD_CLAIM, PROVISIONED, inject_imex_claim
except Exception:  # registry unavailable → never block a deploy
    sys.stdout.write(sys.stdin.read())
    sys.exit(0)


def main() -> None:
    content = sys.stdin.read()
    state = (os.environ.get("NVLINK_MULTICAST_IMEX") or "").strip()
    rct = (os.environ.get("IMEX_CLAIM_TEMPLATE") or "").strip()

    # Only inject when the cluster actually has a provisioned ComputeDomain channel to claim.
    if state != PROVISIONED or not rct:
        sys.stdout.write(content)
        return

    try:
        docs = [d for d in yaml.safe_load_all(content) if d is not None]
        n_wired = 0
        for doc in docs:
            _, n = inject_imex_claim(doc, pod_claim=IMEX_POD_CLAIM, channel_rct=rct)
            n_wired += n
        if n_wired:
            print(
                f"merge_imex_claim: wired ComputeDomain channel claim '{IMEX_POD_CLAIM}' "
                f"(template {rct}) into {n_wired} GPU container(s) → FlashInfer allreduce_rms fusion enabled",
                file=sys.stderr,
            )
        sys.stdout.write(yaml.dump_all(docs, default_flow_style=False, allow_unicode=True))
    except Exception as e:
        print(
            f"merge_imex_claim: warning — injection failed ({e}), applying unmodified",
            file=sys.stderr,
        )
        sys.stdout.write(content)


if __name__ == "__main__":
    main()
