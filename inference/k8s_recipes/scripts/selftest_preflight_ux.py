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

"""selftest_preflight_ux.py — unit tests for the preflight UX helpers.

Covers ONLY the pure, cluster-free functions in preflight.py:
  parse_args          — 0.6 flag parsing (--wait-on-resources in any position; 2 positionals)
  gpu_availability    — the DRA-aware free-GPU accounting, over fake kubectl node/pod/claim dicts
  pick_single_node    — single-node placement verdict (re-used by the wait poll)
  place_disagg        — greedy cross-node disagg placement verdict (re-used by the wait poll)
  gpu_resource_summary — the 📊 resources line
  fix_line            — 0.7 copy-pasteable fix-hint formatting

No kubectl, no cluster. Run with `python3 scripts/selftest_preflight_ux.py` or via `make test`.
Exit 0 = all checks pass.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load_preflight():
    """Import preflight.py without executing main() (guarded by __main__)."""
    spec = importlib.util.spec_from_file_location("preflight", SCRIPTS / "preflight.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pf = _load_preflight()


# ── fixture builders ──────────────────────────────────────────────────────────
def node(name, gpus, arch="amd64", taints=None):
    return {
        "metadata": {"name": name, "labels": {"kubernetes.io/arch": arch}},
        "status": {"allocatable": {"nvidia.com/gpu": str(gpus)}},
        "spec": {"taints": taints or []},
    }


def taint(key, effect="NoSchedule"):
    return {"key": key, "effect": effect}


def pod(name, node_name, gpus=0, phase="Running", ns="bench", init_gpus=0):
    spec = {"nodeName": node_name, "containers": [], "initContainers": []}
    if gpus:
        spec["containers"] = [{"resources": {"requests": {"nvidia.com/gpu": str(gpus)}}}]
    if init_gpus:
        spec["initContainers"] = [{"resources": {"requests": {"nvidia.com/gpu": str(init_gpus)}}}]
    return {
        "metadata": {"namespace": ns, "name": name},
        "status": {"phase": phase},
        "spec": spec,
    }


def dra_claim(name, pod_name, ndev=1, ns="bench"):
    return {
        "metadata": {"namespace": ns, "name": name},
        "status": {
            "allocation": {"devices": {"results": [{"driver": "gpu.nvidia.com", "pool": "gpu-pool"}] * ndev}},
            "reservedFor": [{"name": pod_name}],
        },
    }


def compute_domain_claim(name, pod_name, ndev=1, ns="gpu-operator"):
    """A gpu-operator ComputeDomain (NVLink/IMEX) claim — NOT a GPU. Its JSON contains 'nvidia'
    but its driver is compute-domain.nvidia.com (no 'gpu' token). Must NOT be counted as GPU usage.
    """
    return {
        "metadata": {"namespace": ns, "name": name},
        "status": {
            "allocation": {
                "devices": {"results": [{"driver": "compute-domain.nvidia.com", "pool": "imex-channel"}] * ndev}
            },
            "reservedFor": [{"name": pod_name}],
        },
    }


# ── 0.6 parse_args ────────────────────────────────────────────────────────────
class TestParseArgs(unittest.TestCase):
    def test_positionals_only(self):
        cell, prof, opts = pf.parse_args(["recipes/x", "b200"])
        self.assertEqual((cell, prof), ("recipes/x", "b200"))
        self.assertFalse(opts["wait_on_resources"])

    def test_flag_trailing(self):
        cell, prof, opts = pf.parse_args(["recipes/x", "b200", "--wait-on-resources"])
        self.assertEqual((cell, prof), ("recipes/x", "b200"))
        self.assertTrue(opts["wait_on_resources"])

    def test_flag_leading(self):
        cell, prof, opts = pf.parse_args(["--wait-on-resources", "recipes/x", "b200"])
        self.assertEqual((cell, prof), ("recipes/x", "b200"))
        self.assertTrue(opts["wait_on_resources"])

    def test_flag_between_positionals(self):
        cell, prof, opts = pf.parse_args(["recipes/x", "--wait-on-resources", "b200"])
        self.assertEqual((cell, prof), ("recipes/x", "b200"))
        self.assertTrue(opts["wait_on_resources"])

    def test_too_few_positionals_raises(self):
        with self.assertRaises(ValueError):
            pf.parse_args(["recipes/x"])

    def test_too_many_positionals_raises(self):
        with self.assertRaises(ValueError):
            pf.parse_args(["recipes/x", "b200", "extra"])

    def test_unknown_flag_raises(self):
        with self.assertRaises(ValueError):
            pf.parse_args(["recipes/x", "b200", "--nope"])

    # ── --stage-only (install-time CPU-only mode) ──
    # install STAGES a cell for a run LATER, so live GPU availability is not a precondition: the
    # free-GPU checks must degrade FAIL→WARN and the nvlink-p2p live probe (which schedules a real GPU
    # pod) must be skipped. Otherwise a multi-GPU cell is un-installable on a busy shared cluster.
    def test_stage_only_defaults_false(self):
        _, _, opts = pf.parse_args(["recipes/x", "b200"])
        self.assertFalse(opts["stage_only"])

    def test_stage_only_parsed_any_position(self):
        for argv in (
            ["recipes/x", "b200", "--stage-only"],
            ["--stage-only", "recipes/x", "b200"],
            ["recipes/x", "--stage-only", "b200"],
        ):
            cell, prof, opts = pf.parse_args(argv)
            self.assertEqual((cell, prof), ("recipes/x", "b200"), argv)
            self.assertTrue(opts["stage_only"], argv)

    def test_stage_only_composes_with_wait_on_resources(self):
        _, _, opts = pf.parse_args(["recipes/x", "b200", "--stage-only", "--wait-on-resources"])
        self.assertTrue(opts["stage_only"] and opts["wait_on_resources"])

    def test_warn_does_not_set_failure_rc(self):
        """The FAIL→WARN capacity downgrade must not fail the run (WARNs are advisory)."""
        pf.rc_fail = 0
        pf.line("WARN", "no schedulable node with 8 free GPUs (--stage-only)")
        self.assertEqual(pf.rc_fail, 0)
        pf.line("FAIL", "a real config failure")
        self.assertEqual(pf.rc_fail, 1)
        pf.rc_fail = 0


# ── install wires --stage-only into its preflight invocation ──────────────────
class TestInstallUsesStageOnly(unittest.TestCase):
    def test_install_passes_stage_only(self):
        src = (SCRIPTS / "install.py").read_text()
        self.assertIn(
            '"--stage-only"',
            src,
            "install.py must invoke preflight with --stage-only so staging never holds GPUs "
            "nor hard-fails on busy-cluster capacity",
        )


# ── 0.6 gpu_availability (DRA-aware accounting) ───────────────────────────────
class TestGpuAvailability(unittest.TestCase):
    def test_device_plugin_requests_counted(self):
        nodes = [node("n1", 8)]
        pods = [pod("p1", "n1", gpus=2), pod("p2", "n1", gpus=3)]
        a = pf.gpu_availability(nodes, pods, [], tolerated=set(), want_arch="")
        self.assertEqual(a["used"], {"n1": 5})
        self.assertEqual(a["schedulable"], [["n1", 3]])
        self.assertEqual(a["dra_gpus"], 0)

    def test_init_container_requests_counted(self):
        a = pf.gpu_availability(
            [node("n1", 8)],
            [pod("p1", "n1", init_gpus=4)],
            [],
            tolerated=set(),
            want_arch="",
        )
        self.assertEqual(a["used"], {"n1": 4})
        self.assertEqual(a["schedulable"], [["n1", 4]])

    def test_terminal_pods_ignored(self):
        pods = [
            pod("done", "n1", gpus=8, phase="Succeeded"),
            pod("bad", "n1", gpus=8, phase="Failed"),
        ]
        a = pf.gpu_availability([node("n1", 8)], pods, [], tolerated=set(), want_arch="")
        self.assertEqual(a["used"], {})
        self.assertEqual(a["schedulable"], [["n1", 8]])

    def test_dra_claims_add_to_used(self):
        # p1 holds a DRA GPU (no container request); accounting must still see n1 as 1-in-use.
        nodes = [node("n1", 8)]
        pods = [pod("p1", "n1", gpus=0)]
        claims = [dra_claim("c1", "p1", ndev=2)]
        a = pf.gpu_availability(nodes, pods, claims, tolerated=set(), want_arch="")
        self.assertEqual(a["dra_gpus"], 2)
        self.assertEqual(a["used"], {"n1": 2})
        self.assertEqual(a["schedulable"], [["n1", 6]])

    def test_compute_domain_claims_not_counted(self):
        # A node with 4 GPUs, all in use via device-plugin, PLUS gpu-operator ComputeDomain (IMEX)
        # claims reserved for the same pod. The IMEX claims must NOT inflate `used` past allocatable
        # (the `6 used / 4 alloc` bug) — only real GPU claims count.
        nodes = [node("n1", 4)]
        pods = [pod("p1", "n1", gpus=4)]
        claims = [
            compute_domain_claim("cd1", "p1", ndev=2),
            compute_domain_claim("cd2", "p1", ndev=2),
        ]
        a = pf.gpu_availability(nodes, pods, claims, tolerated=set(), want_arch="")
        self.assertEqual(a["dra_gpus"], 0)  # no GPU claimed via DRA here
        self.assertEqual(a["used"], {"n1": 4})  # not 8 — ComputeDomain excluded
        self.assertEqual(a["schedulable"], [["n1", 0]])

    def test_mixed_gpu_and_compute_domain_claims(self):
        # Real GPU DRA claim (counts) alongside a ComputeDomain claim (does not).
        nodes = [node("n1", 8)]
        pods = [pod("p1", "n1", gpus=0)]
        claims = [
            dra_claim("c1", "p1", ndev=2),
            compute_domain_claim("cd1", "p1", ndev=4),
        ]
        a = pf.gpu_availability(nodes, pods, claims, tolerated=set(), want_arch="")
        self.assertEqual(a["dra_gpus"], 2)
        self.assertEqual(a["used"], {"n1": 2})

    def test_blocking_taint_excludes_node(self):
        nodes = [node("n1", 8, taints=[taint("dedicated")])]
        a = pf.gpu_availability(nodes, [], [], tolerated=set(), want_arch="")
        self.assertEqual(a["schedulable"], [])

    def test_tolerated_taint_keeps_node(self):
        nodes = [node("n1", 8, taints=[taint("dedicated")])]
        a = pf.gpu_availability(nodes, [], [], tolerated={"dedicated"}, want_arch="")
        self.assertEqual(a["schedulable"], [["n1", 8]])

    def test_arch_filter_excludes_mismatch(self):
        nodes = [node("n1", 8, arch="arm64"), node("n2", 8, arch="amd64")]
        a = pf.gpu_availability(nodes, [], [], tolerated=set(), want_arch="amd64")
        self.assertEqual(a["schedulable"], [["n2", 8]])
        self.assertEqual(a["node_archs"], {"arm64", "amd64"})


# ── 0.6 placement verdicts (re-used by the wait poll) ─────────────────────────
class TestPlacementVerdicts(unittest.TestCase):
    def test_pick_single_node_found(self):
        self.assertEqual(pf.pick_single_node([["n1", 2], ["n2", 8]], 8), ["n2", 8])

    def test_pick_single_node_none(self):
        self.assertIsNone(pf.pick_single_node([["n1", 2], ["n2", 4]], 8))

    def test_disagg_across_two_nodes(self):
        assigned = pf.place_disagg([["n1", 8], ["n2", 8]], [("prefill", 8), ("decode", 8)])
        self.assertIsNotNone(assigned)
        self.assertEqual({a.split(":")[0] for a in assigned}, {"prefill", "decode"})
        self.assertEqual({a.split(":")[1].split("(")[0] for a in assigned}, {"n1", "n2"})

    def test_disagg_insufficient_returns_none(self):
        self.assertIsNone(pf.place_disagg([["n1", 8]], [("prefill", 8), ("decode", 8)]))

    def test_disagg_does_not_mutate_input(self):
        sched = [["n1", 8], ["n2", 8]]
        pf.place_disagg(sched, [("prefill", 8), ("decode", 8)])
        self.assertEqual(sched, [["n1", 8], ["n2", 8]])  # caller's list untouched


# ── 0.6 resources line ────────────────────────────────────────────────────────
class TestResourceSummary(unittest.TestCase):
    def test_totals_and_biggest_free(self):
        nodes = [node("n1", 8), node("n2", 8)]
        used = {"n1": 6}
        s = pf.gpu_resource_summary(nodes, used, "NVIDIA-B200")
        self.assertIn("2 node(s)", s)
        self.assertIn("16 total", s)
        self.assertIn("6 in use", s)
        self.assertIn("10 free", s)
        self.assertIn("biggest free node: 8", s)


# ── 0.7 fix-hint formatting ───────────────────────────────────────────────────
class TestFixLine(unittest.TestCase):
    def test_contains_marker_and_command(self):
        out = pf.fix_line("kubectl create namespace bench")
        self.assertIn("→ fix:", out)
        self.assertIn("kubectl create namespace bench", out)

    def test_single_line(self):
        self.assertNotIn("\n", pf.fix_line("some cmd"))


class TestClassifyArtifactsSc(unittest.TestCase):
    """B200 round-2: ARTIFACTS_STORAGE_CLASS decision must not FAIL a run when the class is unverifiable
    but the artifacts PVC is already provisioned (RBAC-forbidden lookup or already-Bound PVC).
    """

    def test_bound_pvc_makes_class_moot(self):
        # even if the class lookup would fail, a Bound artifacts PVC settles it — sweep.sh reuses the PVC
        self.assertEqual(
            pf.classify_artifacts_sc("Bound", 1, "Error (NotFound)"),
            ("PASS", "moot-bound"),
        )

    def test_class_exists(self):
        self.assertEqual(pf.classify_artifacts_sc("", 0, ""), ("PASS", "exists"))

    def test_rbac_forbidden_is_warn_not_fail(self):
        lvl, reason = pf.classify_artifacts_sc("", 1, "Error from server (Forbidden): storageclasses is forbidden")
        self.assertEqual((lvl, reason), ("WARN", "rbac-forbidden"))

    def test_genuine_not_found_still_fails(self):
        lvl, reason = pf.classify_artifacts_sc(
            "Pending",
            1,
            'Error from server (NotFound): storageclasses "vast" not found',
        )
        self.assertEqual((lvl, reason), ("FAIL", "not-found"))

    def test_bound_beats_forbidden(self):
        # precedence: a Bound PVC wins even over a forbidden class lookup
        self.assertEqual(pf.classify_artifacts_sc("Bound", 1, "forbidden")[0], "PASS")


class TestProfileRdmaSelectors(unittest.TestCase):
    """#14: RDMA node labels come from the PROFILE (cluster fabric identity), so a disagg recipe ports across
    clusters with different RDMA labels without a recipe fork."""

    def test_parses_comma_separated_pairs_sorted(self):
        self.assertEqual(
            pf.profile_rdma_selectors(
                {"RDMA_NODE_SELECTOR": "example.com/rdma=true,feature.node.kubernetes.io/rdma.available=true"}
            ),
            [
                ("example.com/rdma", "true"),
                ("feature.node.kubernetes.io/rdma.available", "true"),
            ],
        )

    def test_strips_quotes_and_whitespace(self):
        self.assertEqual(
            pf.profile_rdma_selectors({"RDMA_NODE_SELECTOR": '  k = "v" '}),
            [("k", "v")],
        )

    def test_unset_or_junk_is_empty(self):
        self.assertEqual(pf.profile_rdma_selectors({}), [])
        self.assertEqual(pf.profile_rdma_selectors({"RDMA_NODE_SELECTOR": "no-equals-here"}), [])


class TestDisaggRdma(unittest.TestCase):
    """Proactive RDMA capability check for disagg recipes (pure parts)."""

    def test_is_disagg(self):
        self.assertTrue(pf.is_disagg({"disagg": {"prefill": {}}}))
        self.assertTrue(pf.is_disagg({"stack": "sglang-disagg"}))
        self.assertFalse(pf.is_disagg({"stack": "vllm-agg"}))

    def test_selectors_extracted_from_rendered(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            cell = Path(td)
            (cell / "rendered").mkdir()
            (cell / "rendered" / "workers.yaml").write_text(
                "spec:\n"
                "      nodeSelector:\n"
                "        nvidia.com/gpu.product: ${GPU_PRODUCT}\n"
                '        feature.node.kubernetes.io/rdma.available: "true"\n'
                '        example.com/rdma-enabled: "true"\n'
                "      containers:\n"
                "        - env:\n"
                '            - { name: UCX_NET_DEVICES, value: "mlx5_0:1" }\n'
            )
            sels = dict(pf.disagg_rdma_selectors(cell))
            self.assertEqual(sels.get("feature.node.kubernetes.io/rdma.available"), "true")
            self.assertEqual(sels.get("example.com/rdma-enabled"), "true")
            self.assertNotIn("nvidia.com/gpu.product", sels)  # not an RDMA label
            self.assertNotIn("name", sels)  # the UCX env line is not a nodeSelector

    def test_node_ok_requires_all_selectors(self):
        sels = [
            ("feature.node.kubernetes.io/rdma.available", "true"),
            ("example.com/rdma-enabled", "true"),
        ]
        full = {
            "feature.node.kubernetes.io/rdma.available": "true",
            "example.com/rdma-enabled": "true",
            "other": "x",
        }
        partial = {"feature.node.kubernetes.io/rdma.available": "true"}
        self.assertTrue(pf.rdma_node_ok(sels, [partial, full]))  # one node matches all
        self.assertFalse(pf.rdma_node_ok(sels, [partial]))  # none match all → proactive FAIL
        self.assertFalse(pf.rdma_node_ok(sels, []))  # no nodes at all


class TestKubeContext(unittest.TestCase):
    """preflight must target the profile's cluster, not the ambient current-context (profile-context regression)."""

    def test_pins_context_when_profile_sets_it(self):
        pf._KUBE_CONTEXT = "ctx-prod"
        try:
            argv = pf._kubectl_argv(["get", "ns"])
        finally:
            pf._KUBE_CONTEXT = ""
        self.assertEqual(argv[:3], ["kubectl", "--context", "ctx-prod"])
        self.assertIn("get", argv)

    def test_no_context_when_unset(self):
        pf._KUBE_CONTEXT = ""
        self.assertNotIn("--context", pf._kubectl_argv(["get", "ns"]))


def main() -> int:
    print("selftest_preflight_ux: unit-testing preflight.py pure helpers …")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite(
        loader.loadTestsFromTestCase(tc)
        for tc in (
            TestParseArgs,
            TestGpuAvailability,
            TestPlacementVerdicts,
            TestResourceSummary,
            TestFixLine,
            TestClassifyArtifactsSc,
            TestProfileRdmaSelectors,
            TestDisaggRdma,
            TestKubeContext,
        )
    )
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    if result.wasSuccessful():
        print("selftest_preflight_ux: ALL CHECKS PASSED")
        return 0
    print("selftest_preflight_ux: FAILURES — see above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
