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

"""selftest_fabric_health.py — unit tests for the NVLink P2P fabric-health parsers in probe_fabric.py.

Covers ONLY the PURE, cluster-free surface (no kubectl, no GPU pod):
  parse_p2p_matrix   — `nvidia-smi topo -p2p rw` GPU×GPU matrix → healthy | disabled | unknown
  parse_fabric_health— `nvidia-smi -q` Fabric block → {state, summary, route_unhealthy}
  classify_p2p       — fuse matrix + fabric-health into the recorded NVLINK_P2P fact

Fixtures are the LIVE-captured formats: a HEALTHY pool (OK off-diagonal, Summary=Healthy) and the
BROKEN GB200 NVL72 pool (all-NS matrix + Summary=Unhealthy / Route Unhealthy=True).

Run: `python3 scripts/selftest_fabric_health.py`. Exit 0 = all pass.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("probe_fabric", SCRIPTS / "probe_fabric.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pf = _load()


# ── captured fixtures ────────────────────────────────────────────────────────────
HEALTHY_TOPO = """\
	GPU0	GPU1	GPU2	GPU3
 GPU0	X	OK	OK	OK
 GPU1	OK	X	OK	OK
 GPU2	OK	OK	X	OK
 GPU3	OK	OK	OK	X

Legend:

  X    = Self
  OK   = Status Ok
  NS   = Not Supported
"""

BROKEN_TOPO = """\
	GPU0	GPU1	GPU2	GPU3
 GPU0	X	NS	NS	NS
 GPU1	NS	X	NS	NS
 GPU2	NS	NS	X	NS
 GPU3	NS	NS	NS	X

Legend:

  X    = Self
  NS   = Not Supported
"""

HEALTHY_Q = """\
GPU 00000000:01:00.0
    Product Name                          : NVIDIA GB200
    Performance State                     : P0
    Fabric
        State                             : Completed
        Status                            : Success
        CliqueId                          : 1
        ClusterUUID                       : b2daed23-0000-0000-0000-000000000000
        Health
            Summary                       : Healthy
            Bandwidth                     : Full
            Route Recovery in progress    : False
            Route Unhealthy               : False
            Access Timeout Recovery       : False
"""

BROKEN_Q = """\
GPU 00000000:01:00.0
    Product Name                          : NVIDIA GB200
    Performance State                     : P0
    Fabric
        State                             : Completed
        Status                            : Success
        CliqueId                          : 1
        ClusterUUID                       : b2daed23-0000-0000-0000-000000000000
        Health
            Summary                       : Unhealthy
            Bandwidth                     : Full
            Route Recovery in progress    : False
            Route Unhealthy               : True
            Access Timeout Recovery       : False
"""

# A TRANSIENT fabric-manager route-recovery window: State=In Progress AND `Route Recovery in progress: True`
# with a momentarily Unhealthy block — but the topo matrix is still OK (non-NS). Minor B: this must NOT be
# cached as a hard `disabled` (a false positive that would block a run on a fabric that is actually recovering).
RECOVERING_Q = """\
GPU 00000000:01:00.0
    Product Name                          : NVIDIA GB200
    Performance State                     : P0
    Fabric
        State                             : In Progress
        Status                            : Success
        Health
            Summary                       : Unhealthy
            Bandwidth                     : Full
            Route Recovery in progress    : True
            Route Unhealthy               : True
            Access Timeout Recovery       : False
"""


# ── 1. parse_p2p_matrix ───────────────────────────────────────────────────────────
class TestMatrix(unittest.TestCase):
    def test_healthy_ok_matrix(self):
        self.assertEqual(pf.parse_p2p_matrix(HEALTHY_TOPO), "healthy")

    def test_broken_all_ns_matrix(self):
        self.assertEqual(pf.parse_p2p_matrix(BROKEN_TOPO), "disabled")

    def test_legend_ns_word_not_miscounted(self):
        """The legend line ` NS = Not Supported` must NOT flip a healthy matrix to disabled — only rows
        beginning with a GPUn label are matrix cells."""
        self.assertEqual(pf.parse_p2p_matrix(HEALTHY_TOPO), "healthy")

    def test_any_ns_family_token_disables(self):
        for tok in ("NS", "CNS", "GNS", "TNS"):
            topo = f"\tGPU0\tGPU1\t\n GPU0\tX\t{tok}\t\n GPU1\t{tok}\tX\t\n"
            self.assertEqual(pf.parse_p2p_matrix(topo), "disabled", tok)

    def test_empty_is_unknown(self):
        self.assertEqual(pf.parse_p2p_matrix(""), "unknown")
        self.assertEqual(pf.parse_p2p_matrix("no matrix here"), "unknown")


# ── 2. parse_fabric_health ────────────────────────────────────────────────────────
class TestFabricHealth(unittest.TestCase):
    def test_healthy_block(self):
        h = pf.parse_fabric_health(HEALTHY_Q)
        self.assertEqual(h["state"], "Completed")
        self.assertEqual(h["summary"], "Healthy")
        self.assertIs(h["route_unhealthy"], False)

    def test_broken_block_route_unhealthy(self):
        h = pf.parse_fabric_health(BROKEN_Q)
        self.assertEqual(h["summary"], "Unhealthy")
        self.assertIs(h["route_unhealthy"], True)  # the smoking gun
        self.assertIs(h["route_recovery"], False)  # a HARD fault, not a transient recovery window

    def test_recovering_block_captures_route_recovery_and_state(self):
        """Minor B: the transient window is identified by State=In Progress AND Route Recovery in progress."""
        h = pf.parse_fabric_health(RECOVERING_Q)
        self.assertEqual(h["state"], "In Progress")
        self.assertIs(h["route_recovery"], True)
        self.assertEqual(h["summary"], "Unhealthy")

    def test_performance_state_not_captured_as_fabric_state(self):
        """`Performance State : P0` must not be mistaken for the Fabric `State` line."""
        h = pf.parse_fabric_health(HEALTHY_Q)
        self.assertEqual(h["state"], "Completed")

    def test_empty(self):
        h = pf.parse_fabric_health("")
        self.assertEqual(
            h,
            {
                "state": None,
                "summary": None,
                "route_unhealthy": None,
                "route_recovery": None,
            },
        )


# ── 3. classify_p2p (fusion) ──────────────────────────────────────────────────────
class TestClassify(unittest.TestCase):
    def test_healthy_matrix_and_health(self):
        self.assertEqual(pf.classify_p2p("healthy", pf.parse_fabric_health(HEALTHY_Q)), "healthy")

    def test_broken_matrix_and_health(self):
        self.assertEqual(pf.classify_p2p("disabled", pf.parse_fabric_health(BROKEN_Q)), "disabled")

    def test_route_unhealthy_overrides_ok_matrix(self):
        """Even if the topo matrix looked OK, Route Unhealthy=True (NOT mid-recovery) still means disabled."""
        self.assertEqual(
            pf.classify_p2p("healthy", {"summary": "Unhealthy", "route_unhealthy": True}),
            "disabled",
        )

    def test_recovering_window_with_ok_matrix_is_unknown_not_disabled(self):
        """Minor B: a momentary route-recovery (In Progress / Route Recovery in progress) with an OK topo
        matrix is a false positive — classify as UNKNOWN, never cache a hard `disabled`.
        """
        self.assertEqual(pf.classify_p2p("healthy", pf.parse_fabric_health(RECOVERING_Q)), "unknown")

    def test_ns_matrix_still_disabled_even_mid_recovery(self):
        """The topo matrix NS is the REAL signal — a genuinely-off fabric is `disabled` even while the
        fabric-manager reports it is recovering (corroborated by the matrix)."""
        self.assertEqual(
            pf.classify_p2p("disabled", pf.parse_fabric_health(RECOVERING_Q)),
            "disabled",
        )

    def test_unknown_matrix_but_healthy_fabric(self):
        self.assertEqual(
            pf.classify_p2p("unknown", {"summary": "Healthy", "route_unhealthy": False}),
            "healthy",
        )

    def test_all_unknown(self):
        self.assertEqual(pf.classify_p2p("unknown", {}), "unknown")


# ── 4. GPU probe pod placement and not-ready diagnostics ─────────────────────────
# Let the scheduler select a compatible GPU node and include the scheduling reason
# when the probe cannot become ready. These tests use a fake kubectl client.
class _FakeKubectl:
    """Records kubectl argv and replays programmed (rc, stdout, stderr) responses keyed by a verb match."""

    def __init__(self, wait_rc=0, pod_json="{}"):
        self.calls = []
        self.wait_rc = wait_rc
        self.pod_json = pod_json

    def __call__(self, args, timeout=30):
        self.calls.append(list(args))
        if "wait" in args:
            return (self.wait_rc, "", "" if self.wait_rc == 0 else "timed out")
        if "get" in args and "pod" in args and "-o" in args and "json" in args:
            return (0, self.pod_json, "")
        return (0, "", "")

    def run_spec(self):
        """Return the parsed spec dict from the captured `kubectl run --overrides=...` call."""
        import json as _json

        for c in self.calls:
            for a in c:
                if isinstance(a, str) and a.startswith("--overrides="):
                    return _json.loads(a[len("--overrides=") :])["spec"]
        return None


# Live-captured busy-node admission rejection (nodeName-pin bug signature) and a saturated-pool Pending.
ADMISSION_REJECT_JSON = (
    '{"status": {"phase": "Failed", "reason": "UnexpectedAdmissionError",'
    ' "message": "Pod was rejected: Allocate failed due to requested number of devices unavailable'
    ' for nvidia.com/gpu. Requested: 2, Available: 0, which is unexpected",'
    ' "conditions": [{"type": "PodScheduled", "status": "True"}]}}'
)
UNSCHEDULABLE_JSON = (
    '{"status": {"phase": "Pending", "conditions": ['
    '{"type": "PodScheduled", "status": "False", "reason": "Unschedulable",'
    ' "message": "0/36 nodes are available: 36 Insufficient nvidia.com/gpu."}]}}'
)
IMAGEPULL_JSON = (
    '{"status": {"phase": "Pending", "containerStatuses": [{"state": {"waiting":'
    ' {"reason": "ImagePullBackOff", "message": "Back-off pulling image"}}}]}}'
)


class TestGpuPodSpec(unittest.TestCase):
    def _run(self, fake, **kw):
        orig = pf._kubectl
        pf._kubectl = fake
        try:
            return pf._spawn_gpu_pod("bench", "NVIDIA-GB200", "img:1", **kw)
        finally:
            pf._kubectl = orig

    def test_default_is_scheduler_placed_not_pinned(self):
        fake = _FakeKubectl(wait_rc=0)
        pod, reason = self._run(fake)
        self.assertIsNone(reason)
        spec = fake.run_spec()
        self.assertEqual(spec.get("nodeSelector"), {"nvidia.com/gpu.product": "NVIDIA-GB200"})
        self.assertNotIn("nodeName", spec, "default must NOT pin nodeName (bypasses scheduler gate)")
        self.assertEqual(spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"], "2")
        self.assertEqual(spec["tolerations"], [{"operator": "Exists"}])

    def test_explicit_node_override_pins_and_drops_selector(self):
        fake = _FakeKubectl(wait_rc=0)
        self._run(fake, node_name="node-9")
        spec = fake.run_spec()
        self.assertEqual(spec.get("nodeName"), "node-9")
        self.assertNotIn("nodeSelector", spec)

    def test_pull_secret_and_gpu_count(self):
        fake = _FakeKubectl(wait_rc=0)
        self._run(fake, pull_secret="nvcrio-cred", gpu_count=4)
        spec = fake.run_spec()
        self.assertEqual(spec["imagePullSecrets"], [{"name": "nvcrio-cred"}])
        self.assertEqual(spec["containers"][0]["resources"]["limits"]["nvidia.com/gpu"], "4")

    def test_not_ready_returns_diagnostic_reason(self):
        fake = _FakeKubectl(wait_rc=1, pod_json=ADMISSION_REJECT_JSON)
        pod, reason = self._run(fake)
        self.assertIsNotNone(reason)
        self.assertIn("UnexpectedAdmissionError", reason)
        self.assertIn("devices unavailable", reason)


class TestPodNotReadyReason(unittest.TestCase):
    def _reason(self, pod_json):
        orig = pf._kubectl
        pf._kubectl = _FakeKubectl(pod_json=pod_json)
        try:
            return pf._pod_not_ready_reason("bench", "p")
        finally:
            pf._kubectl = orig

    def test_admission_reject(self):
        r = self._reason(ADMISSION_REJECT_JSON)
        self.assertIn("phase=Failed", r)
        self.assertIn("UnexpectedAdmissionError", r)

    def test_unschedulable_pool_saturated(self):
        r = self._reason(UNSCHEDULABLE_JSON)
        self.assertIn("Unschedulable", r)
        self.assertIn("Insufficient nvidia.com/gpu", r)

    def test_imagepull(self):
        r = self._reason(IMAGEPULL_JSON)
        self.assertIn("ImagePullBackOff", r)


class TestProbeNvlinkSafeDegrade(unittest.TestCase):
    def test_node_override_unknown_carries_error_when_pod_not_ready(self):
        """The single-node diagnostic override safe-degrades to unknown WITH the diagnostic reason."""
        orig = pf._kubectl
        pf._kubectl = _FakeKubectl(wait_rc=1, pod_json=UNSCHEDULABLE_JSON)
        try:
            out = pf.probe_nvlink_p2p("bench", "NVIDIA-GB200", node_name="node-9")
        finally:
            pf._kubectl = orig
        self.assertEqual(out["state"], "unknown")
        self.assertIn("Unschedulable", out.get("error", ""))


# ── 5. clique sweep — per-NVLink-domain probing (the busy-broken-clique fix) ───────
# Fabric health is PER-CLIQUE on NVL72 domains. A single scheduler-placed probe lands on whichever clique
# has free GPUs and MISSES a fault in a busy-but-broken clique (the live GB200 case). So the probe sweeps
# every clique with capacity, reports saturated cliques as `unknown` (never silently passed), and any one
# `disabled` clique makes the whole verdict `disabled`.
def _node(name, clique, alloc=4):
    return {
        "metadata": {"name": name, "labels": {"nvidia.com/gpu.clique": clique}},
        "status": {"allocatable": {"nvidia.com/gpu": str(alloc)}},
    }


class TestCliqueCapacity(unittest.TestCase):
    def _run(self, nodes, used, gpu_count=2):
        of, ou = pf.fetch_gpu_nodes, pf._used_gpus_by_node
        pf.fetch_gpu_nodes = lambda gp: (nodes, None)
        pf._used_gpus_by_node = lambda: used
        try:
            return pf.cliques_with_capacity("NVIDIA-GB200", gpu_count)
        finally:
            pf.fetch_gpu_nodes, pf._used_gpus_by_node = of, ou

    def test_busy_broken_clique_still_probeable_with_2gpu(self):
        """The broken clique is saturated for 4-GPU pods but a node has 2 free — a 2-GPU probe reaches it."""
        nodes = [_node("healthy-a", "CL_HEALTHY"), _node("broken-a", "CL_BROKEN")]
        used = {"healthy-a": 0, "broken-a": 2}  # broken node has 2 free
        probeable, saturated, err = self._run(nodes, used, gpu_count=2)
        self.assertIsNone(err)
        self.assertIn("CL_BROKEN", probeable)
        self.assertIn("CL_HEALTHY", probeable)

    def test_saturated_clique_reported_not_probeable(self):
        nodes = [_node("healthy-a", "CL_HEALTHY"), _node("broken-a", "CL_BROKEN")]
        used = {"healthy-a": 0, "broken-a": 4}  # broken clique fully busy
        probeable, saturated, err = self._run(nodes, used, gpu_count=2)
        self.assertEqual(probeable, ["CL_HEALTHY"])
        self.assertEqual(saturated, [("CL_BROKEN", 0)])

    def test_no_usage_visibility_probes_every_clique(self):
        """No `get pods` RBAC → used empty → free==alloc → optimistically probe both cliques."""
        nodes = [_node("a", "CL1"), _node("b", "CL2")]
        probeable, saturated, err = self._run(nodes, {}, gpu_count=4)
        self.assertEqual(sorted(probeable), ["CL1", "CL2"])
        self.assertEqual(saturated, [])


class TestAggregateCliques(unittest.TestCase):
    def test_any_disabled_clique_disables_whole_pool(self):
        results = [
            {"clique": "CL_HEALTHY", "state": "healthy", "route_unhealthy": False},
            {
                "clique": "CL_BROKEN",
                "state": "disabled",
                "route_unhealthy": True,
                "summary": "Unhealthy",
            },
        ]
        out = pf.aggregate_clique_results(results)
        self.assertEqual(out["state"], "disabled")
        self.assertIs(out["route_unhealthy"], True)
        self.assertIn("CL_BROKEN", out["error"])

    def test_healthy_with_unverified_saturated_clique_notes_it(self):
        results = [
            {"clique": "CL_HEALTHY", "state": "healthy", "route_unhealthy": False},
            {"clique": "CL_BUSY", "state": "unknown", "error": "clique saturated"},
        ]
        out = pf.aggregate_clique_results(results)
        self.assertEqual(out["state"], "healthy")
        self.assertIn("UNVERIFIED", out["error"])
        self.assertIn("CL_BUSY", out["error"])

    def test_all_unknown_stays_unknown_with_reasons(self):
        results = [{"clique": "CL1", "state": "unknown", "error": "saturated"}]
        out = pf.aggregate_clique_results(results)
        self.assertEqual(out["state"], "unknown")
        self.assertIn("CL1", out["error"])


class TestSweepEndToEnd(unittest.TestCase):
    def test_sweep_returns_disabled_when_broken_clique_probed(self):
        """Broken clique is busy (2 free) but reachable by the 2-GPU probe → overall disabled."""
        of, ou, os_, or_, ok = (
            pf.cliques_with_capacity,
            pf._used_gpus_by_node,
            pf._spawn_gpu_pod,
            pf._read_pod_p2p,
            pf._kubectl,
        )
        pf.cliques_with_capacity = lambda gp, gc: (
            ["CL_HEALTHY", "CL_BROKEN"],
            [],
            None,
        )
        pf._spawn_gpu_pod = lambda *a, **k: (f"pod-{k.get('clique')}", None)
        pf._read_pod_p2p = lambda ns, pod: (
            {
                "state": "disabled",
                "matrix_state": "disabled",
                "summary": "Unhealthy",
                "route_unhealthy": True,
            }
            if "CL_BROKEN" in pod
            else {
                "state": "healthy",
                "matrix_state": "healthy",
                "summary": "Healthy",
                "route_unhealthy": False,
            }
        )
        pf._kubectl = lambda *a, **k: (0, "", "")
        try:
            out = pf.probe_nvlink_p2p("bench", "NVIDIA-GB200")
        finally:
            (
                pf.cliques_with_capacity,
                pf._used_gpus_by_node,
                pf._spawn_gpu_pod,
                pf._read_pod_p2p,
                pf._kubectl,
            ) = (
                of,
                ou,
                os_,
                or_,
                ok,
            )
        self.assertEqual(out["state"], "disabled")
        self.assertIs(out["route_unhealthy"], True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
