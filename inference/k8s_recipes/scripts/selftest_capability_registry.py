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

"""selftest_capability_registry.py — unit tests for the recipe-scoped capability registry (Tier 1).

Covers ONLY the pure, cluster-free surface of capability_registry.py:
  requires(recipe)  — the spine: true when the recipe needs the capability, false (silent no-op) otherwise
  probe(facts)      — cluster-fact parse over CANNED facts (no live kubectl)
  gap(recipe,facts) — the guard/degrade decision + action
  evaluate()        — the `required ∩ missing` runner, incl. the recipe-scoping invariant
  relax_serving_env / strip_forced_flashinfer — the apply-time strip that kills the FlashInfer crash class
                      (C1 graceful degrade; also exercised end-to-end through merge_imex_strip.py)
  facts_to_profile_flags / patch_profile_text — the Layer-1 auto-set of no-internet IPs
  gather_facts      — the impure edge, driven by a FAKE krun (canned kubectl JSON)

The headline test is recipe-scoping: a non-fabric aggregated recipe triggers ZERO fabric flags.

Run: `python3 scripts/selftest_capability_registry.py` or via `make test`. Exit 0 = all pass.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location("capability_registry", SCRIPTS / "capability_registry.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass field-type resolution (py3.9 + `from __future__ import annotations`)
    # can find the module via sys.modules[cls.__module__].
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cr = _load()


# ── recipe fixtures ─────────────────────────────────────────────────────────────
def fabric_recipe():
    """GB300 tp8 that FORCES FlashInfer → engages the NVLink fusion (nvlink-imex required)."""
    return {
        "envelope": {
            "gpu_type": "GB300",
            "arch": "arm64",
            "scenario": "llm-perf",
            "provenance": {"image_ref": "nvcr.io/x/vllm@sha256:deadbeef"},
        },
        "serving": {
            "tp": 8,
            "model_repo": "nvidia/Nemotron",
            "model_revision": "183968f8cafe",
            "env": [
                {"name": "VLLM_ATTENTION_BACKEND", "value": "FLASHINFER"},
                {"name": "HF_HUB_OFFLINE", "value": "1"},
            ],
        },
    }


def agg_b200_recipe():
    """Aggregated B200, tp8, NO forced FLASHINFER, internet-allowed llm-perf → NO fabric capabilities."""
    return {
        "envelope": {
            "gpu_type": "B200",
            "arch": "amd64",
            "scenario": "llm-perf",
            "provenance": {"image_ref": "nvcr.io/x/vllm@sha256:b200cafe"},
        },
        "serving": {
            "tp": 8,
            "model_repo": "nvidia/Nemotron",
            "model_revision": "abc123def456",
            "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}],
        },
    }


def no_internet_recipe(no_internet=True):
    r = {
        "envelope": {
            "gpu_type": "GB200",
            "arch": "arm64",
            "scenario": "llm-perf",
            "provenance": {"image_ref": "nvcr.io/x/vllm@sha256:cafe01"},
        },
        "serving": {"tp": 4, "model_repo": "nvidia/Nemotron"},
        "deploy": {"no_internet": no_internet},
    }
    return r


# ── 1. requires() — the spine (true/false) ──────────────────────────────────────
class TestRequires(unittest.TestCase):
    def test_nvlink_true_only_when_forced_tp_gt1_gb(self):
        self.assertTrue(cr._req_nvlink_imex(fabric_recipe()))

    def test_nvlink_false_when_not_forced(self):
        self.assertFalse(cr._req_nvlink_imex(agg_b200_recipe()))

    def test_nvlink_false_when_tp1(self):
        r = fabric_recipe()
        r["serving"]["tp"] = 1
        self.assertFalse(cr._req_nvlink_imex(r))

    def test_nvlink_false_when_not_gb_class(self):
        r = fabric_recipe()
        r["envelope"]["gpu_type"] = "B200"
        self.assertFalse(cr._req_nvlink_imex(r))  # forced FLASHINFER on B200 is NOT the NVLink fusion

    def test_no_internet_true_for_deploy_no_internet(self):
        self.assertTrue(cr._req_no_internet_ips(no_internet_recipe(no_internet=True)))

    def test_no_internet_default_true_when_key_absent(self):
        r = no_internet_recipe()
        del r["deploy"]["no_internet"]
        self.assertTrue(cr._req_no_internet_ips(r))  # template default is no_internet=True

    def test_no_internet_false_when_internet_allowed(self):
        self.assertFalse(cr._req_no_internet_ips(no_internet_recipe(no_internet=False)))

    def test_no_internet_false_for_non_deploy(self):
        self.assertFalse(cr._req_no_internet_ips(agg_b200_recipe()))

    def test_model_revision_true_when_pinned(self):
        self.assertTrue(cr._req_model_revision(fabric_recipe()))

    def test_model_revision_false_when_unpinned(self):
        r = agg_b200_recipe()
        del r["serving"]["model_revision"]
        self.assertFalse(cr._req_model_revision(r))

    def test_image_arch_true_when_arch_declared(self):
        self.assertTrue(cr._req_image_arch(fabric_recipe()))

    def test_image_arch_false_when_no_arch(self):
        r = agg_b200_recipe()
        del r["envelope"]["arch"]
        self.assertFalse(cr._req_image_arch(r))


# ── 2. probe() — present/absent over canned facts ───────────────────────────────
class TestProbe(unittest.TestCase):
    def test_nvlink_provisioned(self):
        self.assertEqual(
            cr._probe_nvlink_imex({"nvlink_imex": {"crd_present": True, "channel_provisioned": True}}),
            cr.PROVISIONED,
        )

    def test_nvlink_available_when_crd_but_unclaimed(self):
        self.assertEqual(
            cr._probe_nvlink_imex({"nvlink_imex": {"crd_present": True, "channel_provisioned": False}}),
            cr.AVAILABLE,
        )

    def test_nvlink_absent(self):
        self.assertEqual(
            cr._probe_nvlink_imex({"nvlink_imex": {"crd_present": False, "channel_provisioned": False}}),
            cr.ABSENT,
        )

    def test_nvlink_unknown_when_no_fact(self):
        self.assertEqual(cr._probe_nvlink_imex({}), cr.UNKNOWN)

    def test_no_internet_available_when_both_ips(self):
        self.assertEqual(
            cr._probe_no_internet_ips(
                {
                    "no_internet": {
                        "dns_ip": "192.168.0.10",
                        "kube_api_ip": "172.16.0.18",
                    }
                }
            ),
            cr.AVAILABLE,
        )

    def test_no_internet_absent_when_missing_one(self):
        self.assertEqual(
            cr._probe_no_internet_ips({"no_internet": {"dns_ip": "192.168.0.10", "kube_api_ip": ""}}),
            cr.ABSENT,
        )

    def test_model_revision_available_when_cache_probed(self):
        self.assertEqual(
            cr._probe_model_revision({"model_cache": {"revisions": ["abc"]}}),
            cr.AVAILABLE,
        )

    def test_model_revision_unknown_when_not_probed(self):
        self.assertEqual(cr._probe_model_revision({}), cr.UNKNOWN)


# ── 3. gap() — the guard/degrade decision + action ──────────────────────────────
class TestGap(unittest.TestCase):
    def test_nvlink_absent_strips_forced_env(self):
        g = cr._gap_nvlink_imex(fabric_recipe(), {}, cr.ABSENT)
        self.assertEqual(g.level, "WARN")
        self.assertEqual(g.action, "strip-forced-env")
        self.assertIn("STRIPPED", g.message)
        self.assertIn("code=800", g.message)

    def test_nvlink_provisioned_passes(self):
        g = cr._gap_nvlink_imex(fabric_recipe(), {}, cr.PROVISIONED)
        self.assertEqual(g.level, "PASS")
        self.assertIsNone(g.action)

    def test_no_internet_absent_fails_with_autowrite(self):
        g = cr._gap_no_internet_ips(
            no_internet_recipe(),
            {"no_internet": {"dns_ip": "", "kube_api_ip": ""}},
            cr.ABSENT,
        )
        self.assertEqual(g.level, "FAIL")
        self.assertEqual(g.action, "auto-write-profile")

    def test_no_internet_available_passes(self):
        facts = {"no_internet": {"dns_ip": "192.168.0.10", "kube_api_ip": "172.16.0.18"}}
        g = cr._gap_no_internet_ips(no_internet_recipe(), facts, cr.AVAILABLE)
        self.assertEqual(g.level, "PASS")

    def test_model_revision_not_cached_warns_and_lists_cached(self):
        facts = {"model_cache": {"revisions": {"nvidia/Nemotron": ["cafef00dbabe"]}}}
        g = cr._gap_model_revision(fabric_recipe(), facts, cr.AVAILABLE)
        self.assertEqual(g.level, "WARN")
        self.assertEqual(g.action, "install-revision")
        self.assertIn("cafef00dbabe", g.message)  # surfaces what IS cached

    def test_model_revision_cached_passes(self):
        facts = {"model_cache": {"revisions": {"nvidia/Nemotron": ["183968f8cafe"]}}}
        g = cr._gap_model_revision(fabric_recipe(), facts, cr.AVAILABLE)
        self.assertEqual(g.level, "PASS")

    def test_image_arch_node_mismatch_fails(self):
        facts = {"node_archs": ["amd64"], "images": None}
        g = cr._gap_image_arch(fabric_recipe(), facts, cr.AVAILABLE)  # recipe arch=arm64, node amd64
        self.assertEqual(g.level, "FAIL")
        self.assertEqual(g.action, "rebuild-image")

    def test_image_arch_manifest_not_covering_fails(self):
        img = "nvcr.io/x/vllm@sha256:deadbeef"
        facts = {
            "node_archs": ["arm64"],
            "images": {img: ["amd64"]},
        }  # image lacks arm64
        g = cr._gap_image_arch(fabric_recipe(), facts, cr.AVAILABLE)
        self.assertEqual(g.level, "FAIL")

    def test_image_arch_covered_passes(self):
        img = "nvcr.io/x/vllm@sha256:deadbeef"
        facts = {"node_archs": ["arm64"], "images": {img: ["arm64", "amd64"]}}
        g = cr._gap_image_arch(fabric_recipe(), facts, cr.AVAILABLE)
        self.assertEqual(g.level, "PASS")

    def test_image_arch_all_uninspectable_is_unknown_not_pass(self):
        """Minor A: the recipe pins image(s) but EVERY one failed registry inspection ({}), while the node
        arch matches. Manifest coverage was NOT verified → WARN/UNKNOWN, never a silent PASS.
        """
        facts = {
            "node_archs": ["arm64"],
            "images": {},
        }  # fabric_recipe pins one image, none inspectable
        g = cr._gap_image_arch(fabric_recipe(), facts, cr.AVAILABLE)
        self.assertEqual(g.level, "WARN")
        self.assertEqual(g.state, cr.UNKNOWN)
        self.assertIn("UNVERIFIED", g.message)

    def test_image_arch_no_pinned_images_still_passes(self):
        """A recipe that pins NO images has nothing to verify — the node-arch match alone is a clean PASS."""
        r = {"envelope": {"arch": "arm64"}, "serving": {}}
        g = cr._gap_image_arch(r, {"node_archs": ["arm64"], "images": {}}, cr.AVAILABLE)
        self.assertEqual(g.level, "PASS")


# ── 4. evaluate() + the RECIPE-SCOPING INVARIANT ────────────────────────────────
class TestRecipeScoping(unittest.TestCase):
    def test_non_fabric_recipe_triggers_zero_fabric_flags(self):
        """The core rule: an aggregated cell that doesn't force FLASHINFER and has no deploy block NEVER
        mentions nvlink-imex or no-internet-ips — no probe, no flag, no WARN."""
        ids = {c.id for c in cr.required_capabilities(agg_b200_recipe())}
        self.assertNotIn("nvlink-imex", ids)
        self.assertNotIn("no-internet-ips", ids)
        # it DOES still check its own images + pinned revision (recipe-relevant, not fabric)
        self.assertIn("image-arch", ids)
        self.assertIn("model-revision-cached", ids)

    def test_non_fabric_evaluate_emits_no_nvlink_gap_even_with_absent_facts(self):
        """Even when the cluster LACKS IMEX, a non-fabric recipe produces no nvlink-imex gap at all."""
        facts = {
            "nvlink_imex": {"crd_present": False, "channel_provisioned": False},
            "no_internet": {"dns_ip": "", "kube_api_ip": ""},
            "node_archs": ["amd64"],
            "model_cache": {"revisions": {"nvidia/Nemotron": ["abc123def456"]}},
        }
        gap_ids = {g.id for g in cr.evaluate(agg_b200_recipe(), facts)}
        self.assertNotIn("nvlink-imex", gap_ids)
        self.assertNotIn("no-internet-ips", gap_ids)

    def test_fabric_recipe_pulls_nvlink_into_required_set(self):
        ids = {c.id for c in cr.required_capabilities(fabric_recipe())}
        self.assertIn("nvlink-imex", ids)

    def test_fabric_recipe_on_imexless_cluster_yields_strip_gap(self):
        facts = {
            "nvlink_imex": {"crd_present": False, "channel_provisioned": False},
            "node_archs": ["arm64"],
            "images": {"nvcr.io/x/vllm@sha256:deadbeef": ["arm64"]},
            "model_cache": {"revisions": {"nvidia/Nemotron": ["183968f8cafe"]}},
        }
        gaps = {g.id: g for g in cr.gaps_only(fabric_recipe(), facts)}
        self.assertIn("nvlink-imex", gaps)
        self.assertEqual(gaps["nvlink-imex"].action, "strip-forced-env")


# ── 5. relax_serving_env — the crash-killer render strip ────────────────────────
class TestRelax(unittest.TestCase):
    def test_absent_strips_flashinfer(self):
        kept, stripped = cr.relax_serving_env(fabric_recipe()["serving"], cr.ABSENT)
        self.assertEqual(stripped, ["VLLM_ATTENTION_BACKEND"])
        self.assertNotIn("VLLM_ATTENTION_BACKEND", [e["name"] for e in kept])
        self.assertIn("HF_HUB_OFFLINE", [e["name"] for e in kept])

    def test_provisioned_keeps_everything(self):
        serving = fabric_recipe()["serving"]
        kept, stripped = cr.relax_serving_env(serving, cr.PROVISIONED)
        self.assertEqual(stripped, [])
        self.assertEqual(len(kept), len(serving["env"]))

    def test_recipe_without_flashinfer_unchanged(self):
        kept, stripped = cr.relax_serving_env(agg_b200_recipe()["serving"], cr.ABSENT)
        self.assertEqual(stripped, [])

    def test_available_strips_flashinfer(self):
        """L1: AVAILABLE (CRD present but pod claim NOT wired) still fails cuMulticastCreate code=800, so it
        must NOT be treated as safe-to-keep — only PROVISIONED keeps FLASHINFER (SATISFIED_STATES).
        """
        self.assertEqual(cr.SATISFIED_STATES, (cr.PROVISIONED,))
        kept, stripped = cr.relax_serving_env(fabric_recipe()["serving"], cr.AVAILABLE)
        self.assertEqual(stripped, ["VLLM_ATTENTION_BACKEND"])

    def test_unknown_strips_flashinfer(self):
        kept, stripped = cr.relax_serving_env(fabric_recipe()["serving"], cr.UNKNOWN)
        self.assertEqual(stripped, ["VLLM_ATTENTION_BACKEND"])


class TestStripForcedFlashinfer(unittest.TestCase):
    """C1: the doc-level apply-time DEGRADE — a FLASHINFER-forcing server on a no-IMEX cluster is stripped."""

    def _server(self):
        return {
            "kind": "Deployment",
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm",
                                "env": [
                                    {
                                        "name": "VLLM_ATTENTION_BACKEND",
                                        "value": "FLASHINFER",
                                    },
                                    {"name": "HF_HUB_OFFLINE", "value": "1"},
                                ],
                                "resources": {"requests": {"nvidia.com/gpu": "4"}},
                            }
                        ]
                    }
                }
            },
        }

    def test_absent_strips_from_gpu_container(self):
        doc, n = cr.strip_forced_flashinfer(self._server(), cr.ABSENT)
        self.assertEqual(n, 1)
        names = [e["name"] for e in doc["spec"]["template"]["spec"]["containers"][0]["env"]]
        self.assertNotIn("VLLM_ATTENTION_BACKEND", names)
        self.assertIn("HF_HUB_OFFLINE", names)  # only the fusion-forcing env is dropped

    def test_available_strips(self):
        _, n = cr.strip_forced_flashinfer(self._server(), cr.AVAILABLE)  # L1: AVAILABLE is not safe
        self.assertEqual(n, 1)

    def test_provisioned_keeps_flashinfer(self):
        doc, n = cr.strip_forced_flashinfer(self._server(), cr.PROVISIONED)
        self.assertEqual(n, 0)
        names = [e["name"] for e in doc["spec"]["template"]["spec"]["containers"][0]["env"]]
        self.assertIn("VLLM_ATTENTION_BACKEND", names)

    def test_non_forcing_container_untouched(self):
        dep = {
            "kind": "Deployment",
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm",
                                "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}],
                                "resources": {"requests": {"nvidia.com/gpu": "1"}},
                            }
                        ]
                    }
                }
            },
        }
        _, n = cr.strip_forced_flashinfer(dep, cr.ABSENT)
        self.assertEqual(n, 0)

    def test_non_deployment_untouched(self):
        _, n = cr.strip_forced_flashinfer({"kind": "Service"}, cr.ABSENT)
        self.assertEqual(n, 0)


# ── 6. Layer-1 auto-set (no-internet IPs) profile write ─────────────────────────
class TestProfileAutoset(unittest.TestCase):
    def test_facts_to_flags_writes_both_ips(self):
        facts = {
            "no_internet": {"dns_ip": "192.168.0.10", "kube_api_ip": "172.16.0.18"},
            "nvlink_imex": {"crd_present": True, "channel_provisioned": False},
        }
        flags = cr.facts_to_profile_flags(facts)
        self.assertEqual(flags["NO_INTERNET_DNS_IP"], "192.168.0.10")
        self.assertEqual(flags["NO_INTERNET_KUBE_API_IP"], "172.16.0.18")
        self.assertEqual(flags["NVLINK_MULTICAST_IMEX"], cr.AVAILABLE)

    def test_facts_to_flags_skips_empty(self):
        flags = cr.facts_to_profile_flags({"no_internet": {"dns_ip": "", "kube_api_ip": ""}})
        self.assertNotIn("NO_INTERNET_DNS_IP", flags)

    def test_patch_profile_text_is_idempotent(self):
        base = 'NAMESPACE="bench"\nGPU_PRODUCT="NVIDIA-GB200"\n'
        flags = {
            "NO_INTERNET_DNS_IP": "192.168.0.10",
            "NO_INTERNET_KUBE_API_IP": "172.16.0.18",
        }
        once = cr.patch_profile_text(base, flags)
        twice = cr.patch_profile_text(once, flags)
        self.assertEqual(once, twice)  # re-running doesn't stack blocks
        self.assertIn('NO_INTERNET_DNS_IP="192.168.0.10"', once)
        self.assertEqual(once.count(cr._BLOCK_HEADER), 1)

    def test_patch_profile_text_updates_stale_block(self):
        base = 'NAMESPACE="bench"\n'
        first = cr.patch_profile_text(base, {"NO_INTERNET_DNS_IP": "1.1.1.1"})
        second = cr.patch_profile_text(first, {"NO_INTERNET_DNS_IP": "2.2.2.2"})
        self.assertIn("2.2.2.2", second)
        self.assertNotIn("1.1.1.1", second)


# ── 7. gather_facts over a FAKE krun (canned kubectl JSON) ───────────────────────
class FakeKubectl:
    """Deterministic krun stub: matches on a substring of the args and returns (rc, stdout, stderr)."""

    def __init__(self, table):
        self.table = table

    def __call__(self, args, timeout=30):
        key = " ".join(args)
        for needle, resp in self.table.items():
            if needle in key:
                return resp
        return (1, "", "no-match")


class TestGather(unittest.TestCase):
    def test_gather_facts_parses_crd_and_ips_and_arch(self):
        krun = FakeKubectl(
            {
                "get crd computedomains": (
                    0,
                    "customresourcedefinition.apiextensions.k8s.io/computedomains.resource.nvidia.com",
                    "",
                ),
                "get svc kube-dns": (0, "192.168.0.10", ""),
                "get endpoints kubernetes": (0, "172.16.0.18", ""),
                "get nodes": (0, "arm64 arm64", ""),
            }
        )
        facts = cr.gather_facts({"GPU_PRODUCT": "NVIDIA-GB200"}, krun)
        self.assertTrue(facts["nvlink_imex"]["crd_present"])
        self.assertEqual(facts["no_internet"]["dns_ip"], "192.168.0.10")
        self.assertEqual(facts["no_internet"]["kube_api_ip"], "172.16.0.18")
        self.assertEqual(facts["node_archs"], ["arm64"])

    def test_gather_facts_safe_degrades_on_probe_failure(self):
        krun = FakeKubectl({})  # everything returns rc=1
        facts = cr.gather_facts({"GPU_PRODUCT": "x"}, krun)
        self.assertFalse(facts["nvlink_imex"]["crd_present"])
        self.assertEqual(facts["no_internet"]["dns_ip"], "")
        self.assertEqual(facts["node_archs"], [])


# ── 8. structural: every entry declares the full contract ───────────────────────
class TestWellFormed(unittest.TestCase):
    def test_registry_wellformed(self):
        self.assertEqual(cr.registry_is_wellformed(), [])

    def test_tiers(self):
        by_id = {e.id: e for e in cr.REGISTRY}
        self.assertEqual(by_id["nvlink-imex"].tier, 2)  # flagship ships Tier-2 auto-provision
        self.assertEqual(by_id["nvlink-imex"].provision_fn, cr.plan_nvlink_imex)
        for other in (
            "no-internet-ips",
            "model-revision-cached",
            "image-arch",
            "nvlink-p2p",
            "model-cache-integrity",
        ):
            self.assertEqual(by_id[other].tier, 1)  # the rest are Tier-1 detect+guard
        self.assertIsNone(by_id["nvlink-p2p"].provision_fn)  # DETECT-ONLY: no auto-provision path
        self.assertIsNone(by_id["model-cache-integrity"].provision_fn)  # detect+block; install is the fix


# ── 9. Tier-2 nvlink-imex auto-provision (ComputeDomain + claim wiring) ──────────
def fabric_recipe_forced():
    """Same as fabric_recipe() but tags the forced env so it reads clearly here."""
    return fabric_recipe()


class TestComputeDomainManifest(unittest.TestCase):
    def test_manifest_is_wellformed_v1beta1(self):
        m = cr.computedomain_manifest("llmb-serving-gb300")
        self.assertEqual(m["apiVersion"], "resource.nvidia.com/v1beta1")
        self.assertEqual(m["kind"], "ComputeDomain")
        self.assertEqual(m["metadata"]["namespace"], "llmb-serving-gb300")
        self.assertEqual(m["metadata"]["labels"]["app.kubernetes.io/managed-by"], "llmb-recipe")
        # spec.channel.resourceClaimTemplate.name is REQUIRED by the live CRD schema (design §4.1).
        self.assertEqual(m["spec"]["channel"]["resourceClaimTemplate"]["name"], cr.IMEX_CHANNEL_RCT)
        self.assertEqual(m["spec"]["channel"]["allocationMode"], "Single")
        self.assertEqual(m["spec"]["numNodes"], 0)  # self-gating (driver default feature gate)

    def test_manifest_name_defaults_shared(self):
        self.assertEqual(cr.computedomain_manifest("ns")["metadata"]["name"], cr.IMEX_CD_NAME)


class TestInjectClaim(unittest.TestCase):
    def _server_deployment(self):
        # A fusion-engaging server: GPU-requesting AND forces FLASHINFER (injection is gated on both).
        return {
            "kind": "Deployment",
            "apiVersion": "apps/v1",
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm",
                                "env": [
                                    {
                                        "name": "VLLM_ATTENTION_BACKEND",
                                        "value": "FLASHINFER",
                                    }
                                ],
                                "resources": {
                                    "requests": {"nvidia.com/gpu": "4"},
                                    "limits": {"nvidia.com/gpu": "4"},
                                },
                            }
                        ]
                    }
                }
            },
        }

    def test_injects_pod_and_container_claim(self):
        doc, n = cr.inject_imex_claim(self._server_deployment())
        self.assertEqual(n, 1)
        pod = doc["spec"]["template"]["spec"]
        self.assertEqual(pod["resourceClaims"][0]["name"], cr.IMEX_POD_CLAIM)
        self.assertEqual(pod["resourceClaims"][0]["resourceClaimTemplateName"], cr.IMEX_CHANNEL_RCT)
        self.assertEqual(pod["containers"][0]["resources"]["claims"][0]["name"], cr.IMEX_POD_CLAIM)

    def test_reference_consistency_pod_claim_matches_container(self):
        """The pod-level claim NAME must equal the container-level claim name (k8s DRA binding contract)."""
        doc, _ = cr.inject_imex_claim(self._server_deployment())
        pod = doc["spec"]["template"]["spec"]
        pod_names = {c["name"] for c in pod["resourceClaims"]}
        ctr_names = {c["name"] for c in pod["containers"][0]["resources"]["claims"]}
        self.assertTrue(ctr_names.issubset(pod_names))

    def test_idempotent(self):
        doc, _ = cr.inject_imex_claim(self._server_deployment())
        doc2, n2 = cr.inject_imex_claim(doc)  # re-inject must not duplicate
        pod = doc2["spec"]["template"]["spec"]
        self.assertEqual(len(pod["resourceClaims"]), 1)
        self.assertEqual(len(pod["containers"][0]["resources"]["claims"]), 1)

    def test_non_deployment_untouched(self):
        svc = {"kind": "Service", "spec": {"ports": []}}
        doc, n = cr.inject_imex_claim(svc)
        self.assertEqual(n, 0)
        self.assertNotIn("resourceClaims", doc.get("spec", {}))

    def test_no_gpu_container_untouched(self):
        dep = {
            "kind": "Deployment",
            "spec": {
                "template": {"spec": {"containers": [{"name": "sidecar", "resources": {"requests": {"cpu": "1"}}}]}}
            },
        }
        doc, n = cr.inject_imex_claim(dep)
        self.assertEqual(n, 0)
        self.assertNotIn("resourceClaims", doc["spec"]["template"]["spec"])

    def test_gpu_container_without_flashinfer_not_injected(self):
        """M1: a GPU container that does NOT force FLASHINFER (tp=1 / non-fusion cell sharing a provisioned
        profile) must NOT get a claim it never needs — injection is recipe-scoped, not profile-scoped.
        """
        dep = {
            "kind": "Deployment",
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "vllm",
                                "env": [{"name": "HF_HUB_OFFLINE", "value": "1"}],  # no FLASHINFER
                                "resources": {"requests": {"nvidia.com/gpu": "1"}},
                            }
                        ]
                    }
                }
            },
        }
        doc, n = cr.inject_imex_claim(dep)
        self.assertEqual(n, 0)
        self.assertNotIn("resourceClaims", doc["spec"]["template"]["spec"])
        self.assertNotIn("claims", doc["spec"]["template"]["spec"]["containers"][0]["resources"])


class TestPlanNvlinkImex(unittest.TestCase):
    def _facts(self, crd, rbac, prov=False):
        return {
            "nvlink_imex": {
                "crd_present": crd,
                "rbac_can_create": rbac,
                "channel_provisioned": prov,
            }
        }

    def test_noop_when_recipe_doesnt_force_fusion(self):
        p = cr.plan_nvlink_imex(agg_b200_recipe(), self._facts(True, True), "ns")
        self.assertEqual(p.action, "noop")
        self.assertEqual(p.tier, 0)
        self.assertEqual(p.manifests, ())

    def test_provision_when_self_serve(self):
        p = cr.plan_nvlink_imex(fabric_recipe(), self._facts(True, True), "llmb-serving-gb300")
        self.assertEqual(p.action, "provision")
        self.assertEqual(p.tier, 2)
        self.assertTrue(p.keep_flashinfer)  # the purpose: FLASHINFER stays ON
        self.assertEqual(len(p.manifests), 1)
        self.assertEqual(p.manifests[0]["kind"], "ComputeDomain")
        self.assertEqual(p.profile_flags["NVLINK_MULTICAST_IMEX"], cr.PROVISIONED)
        self.assertEqual(p.profile_flags["IMEX_CLAIM_TEMPLATE"], cr.IMEX_CHANNEL_RCT)

    def test_degrade_when_no_rbac(self):
        p = cr.plan_nvlink_imex(fabric_recipe(), self._facts(True, False), "gb200-ns")
        self.assertEqual(p.action, "degrade")
        self.assertEqual(p.tier, 1)
        self.assertFalse(p.keep_flashinfer)  # strip FLASHINFER (Tier-1 safe)
        self.assertEqual(p.manifests, ())
        self.assertIn("RBAC", p.message)
        self.assertIn("throughput", p.message.lower())
        self.assertIn("provision-imex", p.message)  # tells the operator how to recover

    def test_degrade_when_no_crd(self):
        p = cr.plan_nvlink_imex(fabric_recipe(), self._facts(False, False), "b200-ns")
        self.assertEqual(p.action, "degrade")
        self.assertIn("machinery", p.message)
        self.assertEqual(p.profile_flags["NVLINK_MULTICAST_IMEX"], cr.ABSENT)

    def test_already_provisioned(self):
        p = cr.plan_nvlink_imex(fabric_recipe(), self._facts(True, True, prov=True), "ns")
        self.assertEqual(p.action, "already-provisioned")
        self.assertTrue(p.keep_flashinfer)
        self.assertEqual(p.manifests, ())

    def test_provision_needs_namespace(self):
        p = cr.plan_nvlink_imex(fabric_recipe(), self._facts(True, True), "")
        self.assertEqual(p.action, "degrade")  # no ns → can't apply → degrade

    def test_provision_plan_dispatch(self):
        p = cr.provision_plan(fabric_recipe(), self._facts(True, True), "ns")
        self.assertEqual(p.action, "provision")
        # a Tier-1 entry has no provisioner → noop
        self.assertEqual(
            cr.provision_plan(no_internet_recipe(), {}, "ns", cap_id="no-internet-ips").action,
            "noop",
        )


class TestGatherRbacFact(unittest.TestCase):
    def test_gather_reports_rbac_and_provisioned(self):
        krun = FakeKubectl(
            {
                "get crd computedomains": (
                    0,
                    "customresourcedefinition.apiextensions.k8s.io/computedomains.resource.nvidia.com",
                    "",
                ),
                "auth can-i create computedomains": (0, "yes", ""),
                "get computedomain llmb-imex": (
                    0,
                    "computedomain.resource.nvidia.com/llmb-imex",
                    "",
                ),
                "get svc kube-dns": (0, "192.168.0.10", ""),
                "get endpoints kubernetes": (0, "172.16.0.18", ""),
                "get nodes": (0, "arm64", ""),
            }
        )
        facts = cr.gather_facts({"GPU_PRODUCT": "NVIDIA-GB300", "NAMESPACE": "llmb-serving-gb300"}, krun)
        self.assertTrue(facts["nvlink_imex"]["crd_present"])
        self.assertTrue(facts["nvlink_imex"]["rbac_can_create"])
        self.assertTrue(facts["nvlink_imex"]["channel_provisioned"])

    def test_gather_rbac_false_when_forbidden(self):
        krun = FakeKubectl(
            {
                "get crd computedomains": (0, "computedomains.resource.nvidia.com", ""),
                # auth can-i returns rc=1 (no) — GB200-style no self-serve
            }
        )
        facts = cr.gather_facts({"GPU_PRODUCT": "x", "NAMESPACE": "gb200-ns"}, krun)
        self.assertTrue(facts["nvlink_imex"]["crd_present"])
        self.assertFalse(facts["nvlink_imex"]["rbac_can_create"])
        self.assertFalse(facts["nvlink_imex"]["channel_provisioned"])

    def test_gather_provisioned_from_profile_flag(self):
        """A profile that already recorded provisioned keeps FLASHINFER even without re-probing the CD object."""
        krun = FakeKubectl({"get crd computedomains": (0, "computedomains.resource.nvidia.com", "")})
        facts = cr.gather_facts({"NAMESPACE": "ns", "NVLINK_MULTICAST_IMEX": cr.PROVISIONED}, krun)
        self.assertTrue(facts["nvlink_imex"]["channel_provisioned"])

    def test_rbac_probe_targets_the_profile_namespace(self):
        """M2: `auth can-i create computedomains` must carry `-n <ns>` so a per-namespace RoleBinding is
        judged against the TARGET namespace, not the caller's default."""
        calls = []

        def recording_krun(args, timeout=30):
            calls.append(list(args))
            if "get crd computedomains" in " ".join(args):
                return (0, "computedomains.resource.nvidia.com", "")
            if "auth can-i" in " ".join(args):
                return (0, "yes", "")
            return (1, "", "")

        cr.gather_facts({"GPU_PRODUCT": "x", "NAMESPACE": "gb300-ns"}, recording_krun)
        cani = next(c for c in calls if "can-i" in c)
        self.assertIn("-n", cani)
        self.assertEqual(cani[cani.index("-n") + 1], "gb300-ns")


# ── 10. static/dry-run: GB300 pareto cell — CD + injected claim reference-consistent ──
class TestGb300ParetoReferenceConsistency(unittest.TestCase):
    """The task's static validation: render the ComputeDomain + inject the ResourceClaim for the GB300
    pareto cell and assert well-formed v1beta1 + reference-consistent (pod claim ↔ ComputeDomain RCT).
    """

    CELL = SCRIPTS.parent / "scripts/fixtures/llm_perf_cells/nemotron-ultra-3-gb300-vllm-agg-pareto"

    def test_gb300_pareto_engages_fusion(self):
        import yaml

        recipe = yaml.safe_load((self.CELL / "recipe.yaml").read_text())
        self.assertTrue(cr._req_nvlink_imex(recipe))  # forced FLASHINFER, tp=4, GB300 → fusion engaged

    def test_cd_and_claim_reference_consistent(self):
        import re
        import yaml

        ns = "llmb-serving-gb300"
        cd = cr.computedomain_manifest(ns)
        # merge_imex_claim runs AFTER envsubst in the deploy.sh pipeline; mimic that so the ${VARS} that are
        # bare inside flow mappings parse as YAML (the injection itself never depends on their values).
        raw = (self.CELL / "rendered/server.yaml").read_text()
        substituted = re.sub(r"\$\{[A-Z_][A-Z0-9_]*\}", "placeholder", raw)
        server_docs = [d for d in yaml.safe_load_all(substituted) if isinstance(d, dict)]
        dep = next(d for d in server_docs if d.get("kind") == "Deployment")
        _, n = cr.inject_imex_claim(dep)
        self.assertGreaterEqual(n, 1)  # the vllm GPU container got wired
        pod = dep["spec"]["template"]["spec"]
        # reference consistency: the pod claims exactly the ResourceClaimTemplate the ComputeDomain generates.
        rct_from_cd = cd["spec"]["channel"]["resourceClaimTemplate"]["name"]
        self.assertEqual(pod["resourceClaims"][0]["resourceClaimTemplateName"], rct_from_cd)
        self.assertEqual(
            pod["containers"][0]["resources"]["claims"][0]["name"],
            pod["resourceClaims"][0]["name"],
        )


# ── 11. apply-time patcher merge_imex_claim.py (deploy.sh pipeline stage) ────────
# A fusion-engaging server manifest (GPU + forced FLASHINFER) — what the apply-time patchers see on stdin.
_FUSION_DEPLOY_YAML = (
    "kind: Deployment\n"
    "spec:\n  template:\n    spec:\n      containers:\n      - name: vllm\n"
    "        env:\n        - name: VLLM_ATTENTION_BACKEND\n          value: FLASHINFER\n"
    '        - name: HF_HUB_OFFLINE\n          value: "1"\n'
    '        resources:\n          requests:\n            nvidia.com/gpu: "4"\n'
)


class TestMergeImexClaim(unittest.TestCase):
    def _run(self, yaml_in, env):
        import subprocess

        p = subprocess.run(
            [sys.executable, str(SCRIPTS / "merge_imex_claim.py")],
            input=yaml_in,
            capture_output=True,
            text=True,
            env={**_os_environ(), **env},
        )
        return p.stdout, p.stderr

    def test_passthrough_when_unprovisioned(self):
        out, _ = self._run(
            _FUSION_DEPLOY_YAML,
            {
                "NVLINK_MULTICAST_IMEX": "absent",
                "IMEX_CLAIM_TEMPLATE": "llmb-imex-channel",
            },
        )
        self.assertNotIn("resourceClaims", out)

    def test_injects_when_provisioned(self):
        out, err = self._run(
            _FUSION_DEPLOY_YAML,
            {
                "NVLINK_MULTICAST_IMEX": "provisioned",
                "IMEX_CLAIM_TEMPLATE": "llmb-imex-channel",
            },
        )
        self.assertIn("resourceClaims", out)
        self.assertIn("llmb-imex-channel", out)
        self.assertIn("imex-channel", out)

    def test_passthrough_when_no_template(self):
        out, _ = self._run(_FUSION_DEPLOY_YAML, {"NVLINK_MULTICAST_IMEX": "provisioned"})
        self.assertNotIn("resourceClaims", out)


class TestMergeImexStrip(unittest.TestCase):
    """C1 end-to-end: the apply-time strip stage that delivers the graceful degrade."""

    def _run(self, yaml_in, env):
        import subprocess

        p = subprocess.run(
            [sys.executable, str(SCRIPTS / "merge_imex_strip.py")],
            input=yaml_in,
            capture_output=True,
            text=True,
            env={**_os_environ(), **env},
        )
        return p.stdout, p.stderr

    def test_strips_flashinfer_when_unprovisioned(self):
        """The headline degrade proof: a FLASHINFER-forcing cell resolved to a no-IMEX cluster deploys
        WITHOUT forced FLASHINFER, so it can't CrashLoop on cuMulticastCreate code=800.
        """
        out, err = self._run(_FUSION_DEPLOY_YAML, {"NVLINK_MULTICAST_IMEX": "absent"})
        self.assertNotIn("FLASHINFER", out)
        self.assertIn("HF_HUB_OFFLINE", out)  # unrelated env survives
        self.assertIn("stripped", err.lower())

    def test_strips_when_available(self):  # L1: AVAILABLE is not safe-to-keep
        out, _ = self._run(_FUSION_DEPLOY_YAML, {"NVLINK_MULTICAST_IMEX": "available"})
        self.assertNotIn("FLASHINFER", out)

    def test_strips_when_state_unset(self):
        out, _ = self._run(_FUSION_DEPLOY_YAML, {})
        self.assertNotIn("FLASHINFER", out)

    def test_keeps_flashinfer_when_provisioned(self):
        out, _ = self._run(_FUSION_DEPLOY_YAML, {"NVLINK_MULTICAST_IMEX": "provisioned"})
        self.assertIn("FLASHINFER", out)

    def test_claim_then_strip_are_mutually_exclusive(self):
        """The two patchers chained (deploy.sh order): provisioned → claim injected + FLASHINFER kept."""
        import subprocess

        env = {
            **_os_environ(),
            "NVLINK_MULTICAST_IMEX": "provisioned",
            "IMEX_CLAIM_TEMPLATE": "llmb-imex-channel",
        }
        claim = subprocess.run(
            [sys.executable, str(SCRIPTS / "merge_imex_claim.py")],
            input=_FUSION_DEPLOY_YAML,
            capture_output=True,
            text=True,
            env=env,
        )
        strip = subprocess.run(
            [sys.executable, str(SCRIPTS / "merge_imex_strip.py")],
            input=claim.stdout,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertIn("FLASHINFER", strip.stdout)  # kept (provisioned)
        self.assertIn("resourceClaims", strip.stdout)  # and the claim is wired


# ── 12. nvlink-p2p — the pre-run fabric-health gate (DETECT-ONLY, scenario-aware severity) ──────────
def behavior_recipe():
    """llm-perf net_behavior_score run, tp=4 → engages P2P but the headline is PERF-INVARIANT."""
    return {
        "envelope": {
            "gpu_type": "GB200",
            "arch": "arm64",
            "scenario": "llm-perf",
            "goal": "agentic-behavior",
            "exemplar": {"metric": "net_behavior_score"},
            "provenance": {"image_ref": "nvcr.io/x/vllm@sha256:beh"},
        },
        "serving": {"tp": 4, "model_repo": "nvidia/Nemotron"},
        "deploy": {"no_internet": True},
    }


def goodput_recipe():
    """llm-perf GOODPUT run, tp=4 → P2P-sensitive (throughput-derived headline)."""
    return {
        "envelope": {
            "gpu_type": "GB300",
            "arch": "arm64",
            "scenario": "llm-perf",
            "exemplar": {"metric": "max_goodput"},
            "provenance": {"image_ref": "nvcr.io/x/vllm@sha256:gp"},
        },
        "serving": {"tp": 4, "model_repo": "nvidia/Nemotron"},
        "deploy": {"no_internet": True},
    }


def disagg_recipe():
    """A 1P1D disagg llm-perf recipe: top-level tp may be low but a role tp>1 still engages P2P."""
    return {
        "envelope": {
            "gpu_type": "B200",
            "arch": "amd64",
            "scenario": "llm-perf",
            "provenance": {"image_ref": "nvcr.io/x/sglang@sha256:d"},
        },
        "serving": {"tp": 1, "disagg": {"prefill": {"tp": 8}, "decode": {"tp": 8}}},
    }


class TestNvlinkP2pRequires(unittest.TestCase):
    def test_required_when_tp_gt1(self):
        self.assertTrue(cr._req_nvlink_p2p(fabric_recipe()))  # llm-perf tp=8
        self.assertTrue(cr._req_nvlink_p2p(behavior_recipe()))  # behavior tp=4
        self.assertTrue(cr._req_nvlink_p2p(disagg_recipe()))  # disagg role tp=8 even though serving.tp=1

    def test_not_required_when_tp1(self):
        r = fabric_recipe()
        r["serving"]["tp"] = 1
        self.assertFalse(cr._req_nvlink_p2p(r))  # tp=1 never crosses the P2P fabric

    def test_not_gb_gated(self):
        """Unlike nvlink-imex, P2P health matters on B200 too (not only GB-class)."""
        self.assertTrue(cr._req_nvlink_p2p(agg_b200_recipe()))  # B200 tp=8


class TestNvlinkP2pProbe(unittest.TestCase):
    def test_healthy_available(self):
        self.assertEqual(cr._probe_nvlink_p2p({"nvlink_p2p": {"state": "healthy"}}), cr.AVAILABLE)

    def test_disabled_absent(self):
        self.assertEqual(cr._probe_nvlink_p2p({"nvlink_p2p": {"state": "disabled"}}), cr.ABSENT)

    def test_unknown_when_no_fact_or_unknown_state(self):
        self.assertEqual(cr._probe_nvlink_p2p({}), cr.UNKNOWN)
        self.assertEqual(cr._probe_nvlink_p2p({"nvlink_p2p": {"state": "unknown"}}), cr.UNKNOWN)


class TestNvlinkP2pGapSeverity(unittest.TestCase):
    """The purpose: severity is SCENARIO-AWARE — WARN for a perf-invariant capability run, FAIL for a
    throughput/goodput run, so we never burn a run producing invalid numbers on a dead fabric.
    """

    DISABLED = {"nvlink_p2p": {"state": "disabled", "route_unhealthy": True}}

    def test_perf_run_disabled_fails(self):
        g = cr._gap_nvlink_p2p(fabric_recipe(), self.DISABLED, cr.ABSENT)  # llm-perf
        self.assertEqual(g.level, "FAIL")
        self.assertIn("INVALID", g.message)
        self.assertIn("fabricmanager", (g.fix + g.message).lower())  # names the fix owner/tool
        self.assertIn("cluster-admin", g.fix.lower())

    def test_goodput_run_disabled_fails(self):
        g = cr._gap_nvlink_p2p(goodput_recipe(), self.DISABLED, cr.ABSENT)  # goodput
        self.assertEqual(g.level, "FAIL")

    def test_behavior_run_disabled_only_warns(self):
        g = cr._gap_nvlink_p2p(behavior_recipe(), self.DISABLED, cr.ABSENT)  # behavior net-score
        self.assertEqual(g.level, "WARN")
        self.assertIn("NCCL_IGNORE_DISABLED_P2P=1", g.message)
        self.assertIn("VALID", g.message)  # capability result stays valid

    def test_disabled_message_blames_cluster_not_recipe(self):
        g = cr._gap_nvlink_p2p(fabric_recipe(), self.DISABLED, cr.ABSENT)
        self.assertIn("CLUSTER FABRIC fault", g.message)
        self.assertIn("NOT a recipe problem", g.message)
        self.assertIn("Route Unhealthy", g.message)  # route-health detail surfaced

    def test_healthy_passes(self):
        g = cr._gap_nvlink_p2p(fabric_recipe(), {"nvlink_p2p": {"state": "healthy"}}, cr.AVAILABLE)
        self.assertEqual(g.level, "PASS")

    def test_unknown_warns_safe_degrade(self):
        g = cr._gap_nvlink_p2p(fabric_recipe(), {}, cr.UNKNOWN)
        self.assertEqual(g.level, "WARN")


class TestNvlinkP2pScoping(unittest.TestCase):
    def test_tp1_recipe_produces_no_p2p_gap(self):
        r = fabric_recipe()
        r["serving"]["tp"] = 1
        ids = {c.id for c in cr.required_capabilities(r)}
        self.assertNotIn("nvlink-p2p", ids)

    def test_perf_recipe_on_disabled_fabric_yields_fail_gap(self):
        facts = {
            "nvlink_p2p": {"state": "disabled", "route_unhealthy": True},
            "nvlink_imex": {"crd_present": False, "channel_provisioned": False},
            "node_archs": ["arm64"],
            "images": {"nvcr.io/x/vllm@sha256:deadbeef": ["arm64"]},
            "model_cache": {"revisions": {"nvidia/Nemotron": ["183968f8cafe"]}},
        }
        gaps = {g.id: g for g in cr.gaps_only(fabric_recipe(), facts)}
        self.assertIn("nvlink-p2p", gaps)
        self.assertEqual(gaps["nvlink-p2p"].level, "FAIL")

    def test_gather_reads_p2p_from_profile(self):
        krun = FakeKubectl({})
        facts = cr.gather_facts({"NVLINK_P2P": "disabled", "NVLINK_P2P_ROUTE_UNHEALTHY": "true"}, krun)
        self.assertEqual(facts["nvlink_p2p"]["state"], "disabled")
        self.assertIs(facts["nvlink_p2p"]["route_unhealthy"], True)

    def test_gather_p2p_unknown_when_profile_silent(self):
        facts = cr.gather_facts({"GPU_PRODUCT": "x"}, FakeKubectl({}))
        self.assertEqual(facts["nvlink_p2p"]["state"], "unknown")


# ── 13. model-cache-integrity — exact server snapshot resolution path ──
class TestModelCachePaths(unittest.TestCase):
    def test_repo_dir_slashes_to_dashes(self):
        self.assertEqual(cr.hf_repo_dir("nvidia/Nemotron-3"), "models--nvidia--Nemotron-3")

    def test_server_snapshot_dir_exact_layout(self):
        p = cr.server_snapshot_dir("nemotron-ultra-3-public/huggingface", "nvidia/Nemotron", "abc123")
        self.assertEqual(
            p,
            "nemotron-ultra-3-public/huggingface/hub/models--nvidia--Nemotron/snapshots/abc123",
        )

    def test_server_snapshot_dir_root_subpath(self):
        # subpath '.' or '' → cache root == mount root (some B200 profiles use '.')
        for sub in (".", "", "/"):
            self.assertEqual(
                cr.server_snapshot_dir(sub, "org/m", "rev"),
                "hub/models--org--m/snapshots/rev",
            )


class TestModelCacheVerdict(unittest.TestCase):
    def _complete(self):
        # A COMPLETE SHARDED SNAPSHOT AS THE PROBE ACTUALLY REPORTS IT: the index was read and names 4
        # shards, all 4 resolve, and all 4 were successfully SIZED. The size fields are not decoration —
        # cache_probe_script always emits SHARD_N/SHARD_MIN/SHARD_MAX, and a report where shards exist but
        # none could be sized means the sizing pass returned nothing, which used to disable every byte
        # guard and fall through to COMPLETE (now UNKNOWN; see test_shards_present_but_unsized_is_unknown).
        return {
            "exists": True,
            "config_json": True,
            "index_json": True,
            "shard_count": 4,
            "required_shards": 4,
            "missing_shards": 0,
            "shard_files": 4,
            "shard_min_bytes": 9,
            "shard_max_bytes": 11,
            "refs_consistent": True,
        }

    def test_complete_available(self):
        self.assertEqual(cr.model_cache_verdict(self._complete()), cr.AVAILABLE)

    def test_missing_dir_absent(self):
        self.assertEqual(cr.model_cache_verdict({"exists": False}), cr.ABSENT)

    def test_incomplete_no_shards_absent(self):
        p = self._complete()
        p["shard_count"] = 0  # the LocalEntryNotFoundError signature
        self.assertEqual(cr.model_cache_verdict(p), cr.ABSENT)

    def test_single_file_model_no_index_available(self):
        # A single-file model (e.g. Qwen3-0.6B): one model.safetensors, NO index.json → still COMPLETE.
        # Requiring index.json unconditionally was a false-negative that blocked every single-safetensors model.
        p = self._complete()
        p.update(index_json=False, shard_count=1, required_shards=0, shard_files=1)
        self.assertEqual(cr.model_cache_verdict(p), cr.AVAILABLE)

    def test_partial_download_missing_referenced_shard_absent(self):
        """Gap 1: config+index+SOME shards present (shard_count>0) but a referenced shard dangles → the
        server still crash-loops on it → ABSENT (the old ≥1-shard heuristic would falsely PASS).
        """
        p = self._complete()
        p["shard_count"] = 3
        p["required_shards"] = 4
        p["missing_shards"] = 1
        self.assertEqual(cr.model_cache_verdict(p), cr.ABSENT)

    def test_all_referenced_shards_resolve_available(self):
        p = self._complete()
        p["required_shards"] = 4
        p["missing_shards"] = 0
        self.assertEqual(cr.model_cache_verdict(p), cr.AVAILABLE)

    def test_index_present_but_unparsed_is_not_available(self):
        """INVERTED DELIBERATELY. This test used to assert the opposite — "index not parsed
        (required_shards==0) → fall back to the ≥1-resolved-shard heuristic (safe-degrade)" — and that
        fallback is the defect, not a degrade. index_json=True means the index file EXISTS;
        required_shards==0 then means the probe enumerated NOTHING from it. Falling back to a bare shard
        COUNT certifies a shard set that was never checked, on the one input where we know the evidence
        failed to read. cache_completeness returned ('complete', '… no shard index (single-file model)')
        for a 113-shard model, and install wrote the PERMANENT completion sentinel on that verdict.

        The genuine single-file case (no index file at all) is unaffected — see
        test_single_file_model_no_index_available directly above, which still PASSES to AVAILABLE.
        """
        p = self._complete()
        p["required_shards"] = 0
        p["missing_shards"] = 0
        self.assertEqual(cr.model_cache_verdict(p), cr.ABSENT)

    def test_shards_present_but_unsized_is_unknown(self):
        """SHARDS>0 with no per-file sizes is the sizing half of the probe returning nothing — not a fact
        about the disk. It silently switched off the plausibility gate, the zero-byte veto and the aggregate
        byte veto together, and the verdict fell through to COMPLETE. UNKNOWN → WARN (safe-degrade), never
        a false AVAILABLE. Both shapes: the post-fix empty and the legacy zeros."""
        for sizes in (
            {"shard_files": None, "shard_min_bytes": None, "shard_max_bytes": None},
            {"shard_files": 0, "shard_min_bytes": 0, "shard_max_bytes": 0},
        ):
            p = self._complete()
            p.update(sizes)
            self.assertEqual(cr.model_cache_verdict(p), cr.UNKNOWN, str(sizes))

    def test_refs_inconsistent_still_available(self):
        """Gap 3: refs/main is orthogonal to the pinned-snapshot load path — a GC'd refs/main must NOT fail
        a complete pinned snapshot."""
        p = self._complete()
        p["refs_consistent"] = False
        self.assertEqual(cr.model_cache_verdict(p), cr.AVAILABLE)

    def test_probe_error_unknown(self):
        self.assertEqual(cr.model_cache_verdict({"probe_error": "pod not ready"}), cr.UNKNOWN)

    def test_none_unknown(self):
        self.assertEqual(cr.model_cache_verdict(None), cr.UNKNOWN)


class TestModelCacheIntegrityEntry(unittest.TestCase):
    def test_required_when_repo_and_revision_pinned(self):
        self.assertTrue(cr._req_model_cache_integrity(fabric_recipe()))

    def test_not_required_when_revision_unpinned(self):
        r = agg_b200_recipe()
        del r["serving"]["model_revision"]  # not-relevant → silent no-op
        self.assertFalse(cr._req_model_cache_integrity(r))

    def test_healthy_passes(self):
        facts = {
            "model_cache_integrity": {
                "exists": True,
                "config_json": True,
                "index_json": True,
                "shard_count": 4,
                "required_shards": 4,
                "missing_shards": 0,
                "shard_files": 4,
                "refs_consistent": True,
                "server_path": ".../snapshots/183968f8cafe",
            }
        }
        g = cr.evaluate(fabric_recipe(), facts, registry=(_entry("model-cache-integrity"),))[0]
        self.assertEqual(g.level, "PASS")

    def test_incomplete_fails_and_blocks(self):
        # No resolved weights (shard_count==0) → INCOMPLETE regardless of index.json → FAIL/block.
        facts = {
            "model_cache_integrity": {
                "exists": True,
                "config_json": True,
                "index_json": False,
                "shard_count": 0,
                "refs_consistent": True,
            }
        }
        g = cr.evaluate(fabric_recipe(), facts, registry=(_entry("model-cache-integrity"),))[0]
        self.assertEqual(g.level, "FAIL")  # never spend a model-load on it
        self.assertEqual(g.action, "install-revision")
        self.assertIn("INCOMPLETE", g.message)
        self.assertIn("safetensors", g.message)  # names the missing weights, not the index
        self.assertIn("CrashLoop", g.message)

    def test_absent_dir_fails(self):
        facts = {"model_cache_integrity": {"exists": False}}
        g = cr.evaluate(fabric_recipe(), facts, registry=(_entry("model-cache-integrity"),))[0]
        self.assertEqual(g.level, "FAIL")
        self.assertIn("ABSENT", g.message)

    def test_partial_shard_fails_and_names_missing_shards(self):
        """Gap 1 end-to-end: a partial download (referenced shard dangling) FAILs and the message names the
        specific index-referenced shards that are missing."""
        facts = {
            "model_cache_integrity": {
                "exists": True,
                "config_json": True,
                "index_json": True,
                "shard_count": 3,
                "required_shards": 4,
                "missing_shards": 1,
                "refs_consistent": True,
            }
        }
        g = cr.evaluate(fabric_recipe(), facts, registry=(_entry("model-cache-integrity"),))[0]
        self.assertEqual(g.level, "FAIL")
        self.assertEqual(g.action, "install-revision")
        self.assertIn("shard", g.message.lower())
        self.assertIn("1 of 4", g.message)

    def test_complete_but_stale_refs_passes(self):
        """Gap 3 end-to-end: a COMPLETE pinned snapshot whose refs/main names a GC'd sha still PASSes —
        refs/main is informational, never the blocker."""
        facts = {
            "model_cache_integrity": {
                "exists": True,
                "config_json": True,
                "index_json": True,
                "shard_count": 4,
                "required_shards": 4,
                "missing_shards": 0,
                "shard_files": 4,
                "refs_consistent": False,
            }
        }
        g = cr.evaluate(fabric_recipe(), facts, registry=(_entry("model-cache-integrity"),))[0]
        self.assertEqual(g.level, "PASS")

    def test_probe_error_warns_safe_degrade(self):
        facts = {"model_cache_integrity": {"probe_error": "mounter pod did not ready"}}
        g = cr.evaluate(fabric_recipe(), facts, registry=(_entry("model-cache-integrity"),))[0]
        self.assertEqual(g.level, "WARN")  # UNKNOWN → never blocks an unmeasured cache
        self.assertIsNone(g.action)

    def test_not_relevant_recipe_is_silent(self):
        r = agg_b200_recipe()
        del r["serving"]["model_revision"]
        ids = {c.id for c in cr.required_capabilities(r)}
        self.assertNotIn("model-cache-integrity", ids)  # no probe, no gap


# ── 14. image-arch manifest coverage — FEED facts["images"] (Gate 3) ────────────────────────────────
class TestArchesFromRawManifest(unittest.TestCase):
    def test_multiarch_index_returns_arches(self):
        idx = {
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {"platform": {"os": "linux", "architecture": "amd64"}},
                {"platform": {"os": "linux", "architecture": "arm64"}},
            ],
        }
        self.assertEqual(cr.arches_from_raw_manifest(idx), ["amd64", "arm64"])

    def test_skips_attestation_and_unknown(self):
        idx = {
            "manifests": [
                {"platform": {"os": "linux", "architecture": "arm64"}},
                {
                    "platform": {"os": "unknown", "architecture": "unknown"},
                    "annotations": {"vnd.docker.reference.type": "attestation-manifest"},
                },
            ]
        }
        self.assertEqual(cr.arches_from_raw_manifest(idx), ["arm64"])

    def test_single_image_manifest_returns_none(self):
        # a bare image manifest carries config+layers, NOT a manifests[] array → arch not here → None
        self.assertIsNone(cr.arches_from_raw_manifest({"mediaType": "…manifest.v2+json", "config": {}, "layers": []}))

    def test_json_string_and_garbage(self):
        import json as _j

        self.assertEqual(
            cr.arches_from_raw_manifest(
                _j.dumps({"manifests": [{"platform": {"os": "linux", "architecture": "arm64"}}]})
            ),
            ["arm64"],
        )
        self.assertIsNone(cr.arches_from_raw_manifest("not json"))


class TestGatherImageArchs(unittest.TestCase):
    def test_maps_ref_to_arches_and_omits_unresolvable(self):
        img = "nvcr.io/x/vllm@sha256:deadbeef"
        recipe = fabric_recipe()
        # a fake inspect: resolves the pinned image, returns None for anything else (safe-degrade → omit)
        inspect = lambda ref: ["arm64", "amd64"] if ref == img else None
        got = cr._gather_image_archs(recipe, {}, inspect)
        self.assertEqual(got, {img: ["amd64", "arm64"]})

    def test_inspect_exception_is_swallowed(self):
        def boom(ref):
            raise RuntimeError("registry down")

        self.assertEqual(cr._gather_image_archs(fabric_recipe(), {}, boom), {})  # UNKNOWN, never a false FAIL

    def test_gather_facts_injects_images_with_fake_inspect(self):
        img = "nvcr.io/x/vllm@sha256:deadbeef"
        krun = FakeKubectl({"get nodes": (0, "arm64", "")})
        facts = cr.gather_facts(
            {"GPU_PRODUCT": "NVIDIA-GB300"},
            krun,
            recipe=fabric_recipe(),
            inspect_image=lambda ref: ["arm64"] if ref == img else None,
        )
        self.assertEqual(facts["images"], {img: ["arm64"]})
        # and the fed facts drive the existing coverage gap: arm64 image on arm64 node → PASS
        g = cr.evaluate(fabric_recipe(), facts, registry=(_entry("image-arch"),))[0]
        self.assertEqual(g.level, "PASS")

    def test_pinned_images_includes_launcher_and_task_pod(self):
        r = {
            "envelope": {"provenance": {"image_ref": "nvcr.io/x/vllm@sha256:a"}},
            "serving": {"launcher_image": "nvcr.io/x/serving-controller@sha256:b"},
            "live": {"task_pod_image": "nvcr.io/x/serving-agent-pod@sha256:c"},
        }
        self.assertEqual(
            cr.pinned_images(r),
            sorted(
                [
                    "nvcr.io/x/vllm@sha256:a",
                    "nvcr.io/x/serving-controller@sha256:b",
                    "nvcr.io/x/serving-agent-pod@sha256:c",
                ]
            ),
        )

    def test_manifest_missing_target_arch_still_fails(self):
        # Gate-3 end-to-end: an image whose manifest list LACKS the node arch is flagged (the fed dict)
        img = "nvcr.io/x/vllm@sha256:deadbeef"
        facts = cr.gather_facts(
            {"GPU_PRODUCT": "NVIDIA-GB300"},
            FakeKubectl({"get nodes": (0, "arm64", "")}),
            recipe=fabric_recipe(),
            inspect_image=lambda ref: ["amd64"],
        )  # amd64-only manifest, node is arm64
        g = cr.evaluate(fabric_recipe(), facts, registry=(_entry("image-arch"),))[0]
        self.assertEqual(g.level, "FAIL")
        self.assertEqual(g.action, "rebuild-image")


# ── 15. image-pull-access — the pre-run CREDENTIAL gate (registry auth probe) ───────────────────────
import base64 as _b64
import json as _json
from urllib.parse import urlparse as _urlparse, parse_qs as _parse_qs

_DRV = "nvcr.io/nvidian/serving-driver@sha256:aabbcc"
_SRV = "nvcr.io/nvidian/vllm@sha256:ddeeff"


def _ngc_dockercfg(user="$oauthtoken", key="NGCKEY"):
    auth = _b64.b64encode(f"{user}:{key}".encode()).decode()
    return _json.dumps({"auths": {"nvcr.io": {"auth": auth}}})


class FakeRegistry:
    """A canned nvcr-style registry: unauthenticated manifest HEAD → 401 Bearer challenge; the proxy_auth
    token endpoint grants a token UNLESS the requested repo scope is in `forbidden` (→ 403); a bearer'd HEAD
    → 200. `network` refs raise (→ UNKNOWN). Deterministic — no real HTTP."""

    def __init__(self, forbidden=(), network=(), token_needs_cred=False):
        self.forbidden = set(forbidden)
        self.network = set(network)
        self.token_needs_cred = token_needs_cred

    def __call__(self, method, url, headers=None):
        headers = headers or {}
        if any(n in url for n in self.network):
            raise OSError("registry unreachable")
        if "/manifests/" in url and not headers.get("Authorization", "").startswith("Bearer"):
            return (
                401,
                {"WWW-Authenticate": 'Bearer realm="https://nvcr.io/proxy_auth",service="nvcr.io"'},
                "",
            )
        if "proxy_auth" in url:
            scope = _parse_qs(_urlparse(url).query).get("scope", [""])[0]
            for repo in self.forbidden:
                if scope == f"repository:{repo}:pull":
                    return 403, {}, ""
            if self.token_needs_cred and not headers.get("Authorization"):
                return 401, {}, ""
            return 200, {}, _json.dumps({"token": "TOK"})
        if "/manifests/" in url:  # bearer'd HEAD
            return 200, {}, ""
        return 500, {}, ""


class TestPullParseHelpers(unittest.TestCase):
    def test_parse_dockerconfig_auth_shape(self):
        auths = cr.parse_dockerconfigjson(_ngc_dockercfg())
        self.assertEqual(auths["nvcr.io"], ("$oauthtoken", "NGCKEY"))

    def test_parse_dockerconfig_userpass_shape_and_host_normalize(self):
        cfg = _json.dumps({"auths": {"https://index.docker.io/v1/": {"username": "u", "password": "p"}}})
        auths = cr.parse_dockerconfigjson(cfg)
        self.assertEqual(auths["docker.io"], ("u", "p"))  # scheme/path stripped, hub alias folded

    def test_parse_dockerconfig_garbage_is_none(self):
        self.assertIsNone(cr.parse_dockerconfigjson("not json"))
        self.assertIsNone(cr.parse_dockerconfigjson(b"\xff\xfe"))
        self.assertIsNone(cr.parse_dockerconfigjson({"no": "auths"}))

    def test_split_image_ref_digest_and_tag(self):
        self.assertEqual(
            cr.split_image_ref(_DRV),
            ("nvcr.io", "nvidian/serving-driver", "sha256:aabbcc"),
        )
        self.assertEqual(
            cr.split_image_ref("nvcr.io/nvidian/vllm:v1"),
            ("nvcr.io", "nvidian/vllm", "v1"),
        )
        self.assertEqual(cr.split_image_ref("busybox:1.36"), ("docker.io", "library/busybox", "1.36"))

    def test_parse_www_authenticate(self):
        c = cr._parse_www_authenticate('Bearer realm="https://nvcr.io/proxy_auth",service="nvcr.io",scope="x"')
        self.assertEqual(c["realm"], "https://nvcr.io/proxy_auth")
        self.assertEqual(c["service"], "nvcr.io")
        self.assertEqual(cr._parse_www_authenticate("Basic realm=x"), {})  # non-Bearer → empty

    def test_registry_cred_for(self):
        auths = cr.parse_dockerconfigjson(_ngc_dockercfg())
        self.assertEqual(cr.registry_cred_for(auths, "nvcr.io"), ("$oauthtoken", "NGCKEY"))
        self.assertIsNone(cr.registry_cred_for(auths, "ghcr.io"))


class TestPullProbe(unittest.TestCase):
    def test_all_pass_is_available(self):
        res = cr.probe_image_pull_access([_DRV, _SRV], _ngc_dockercfg(), FakeRegistry())
        self.assertEqual(res[_DRV]["status"], cr._PULL_PASS)
        self.assertEqual(cr.image_pull_access_verdict(res), cr.AVAILABLE)

    def test_forbidden_repo_is_absent(self):
        # case: the cred authenticates but lacks the serving-driver org → 403 → block
        res = cr.probe_image_pull_access(
            [_DRV, _SRV],
            _ngc_dockercfg(),
            FakeRegistry(forbidden={"nvidian/serving-driver"}),
        )
        self.assertEqual(res[_DRV]["status"], cr._PULL_FORBIDDEN)
        self.assertEqual(res[_SRV]["status"], cr._PULL_PASS)  # different, accessible org still pulls
        self.assertEqual(cr.image_pull_access_verdict(res), cr.ABSENT)

    def test_forbidden_at_manifest_after_token(self):
        # token granted but WITHOUT the repo scope → the bearer'd manifest HEAD 403s → still a gap
        class TokOkManifest403(FakeRegistry):
            def __call__(self, method, url, headers=None):
                headers = headers or {}
                if "/manifests/" in url and headers.get("Authorization", "").startswith("Bearer"):
                    return 403, {}, ""
                return super().__call__(method, url, headers)

        res = cr.probe_image_pull_access([_DRV], _ngc_dockercfg(), TokOkManifest403())
        self.assertEqual(res[_DRV]["status"], cr._PULL_FORBIDDEN)

    def test_network_error_is_unknown(self):
        res = cr.probe_image_pull_access([_DRV], _ngc_dockercfg(), FakeRegistry(network={"serving-driver"}))
        self.assertEqual(res[_DRV]["status"], cr._PULL_UNKNOWN)
        self.assertEqual(cr.image_pull_access_verdict(res), cr.UNKNOWN)  # transient blip never blocks

    def test_public_200_is_pass_without_cred(self):
        # a registry that serves the manifest HEAD 200 unauthenticated (public image) → PASS, no cred needed
        pub = lambda method, url, headers=None: (200, {}, "")
        res = cr.probe_image_pull_access([_DRV], None, pub)
        self.assertEqual(res[_DRV]["status"], cr._PULL_PASS)


class TestPullGap(unittest.TestCase):
    def _recipe(self):
        return {
            "envelope": {"provenance": {"image_digest": _DRV}},
            "serving": {"launcher_image": _SRV},
        }

    def test_forbidden_gap_is_fail_with_fix(self):
        res = cr.probe_image_pull_access(
            [_DRV, _SRV],
            _ngc_dockercfg(),
            FakeRegistry(forbidden={"nvidian/serving-driver"}),
        )
        facts = {"image_pull_access": {"secret": "ngc-registry", "results": res}}
        g = cr.evaluate(self._recipe(), facts, registry=(_entry("image-pull-access"),))[0]
        self.assertEqual(g.level, "FAIL")
        self.assertEqual(g.action, "grant-registry-access")
        self.assertIn("ngc-registry", g.message)  # which secret
        self.assertIn("nvidian/serving-driver", g.message)  # which repo
        self.assertIn("nvidian/serving-driver", g.fix)  # the fix names the org to grant
        self.assertIn("mirror", g.fix.lower())

    def test_all_pass_gap_is_pass(self):
        res = cr.probe_image_pull_access([_DRV, _SRV], _ngc_dockercfg(), FakeRegistry())
        facts = {"image_pull_access": {"secret": "ngc-registry", "results": res}}
        g = cr.evaluate(self._recipe(), facts, registry=(_entry("image-pull-access"),))[0]
        self.assertEqual(g.level, "PASS")

    def test_unknown_gap_is_warn_not_fail(self):
        # network blip / unreadable secret → results=None → WARN (safe-degrade), never a false block
        facts = {"image_pull_access": {"secret": "ngc-registry", "results": None}}
        g = cr.evaluate(self._recipe(), facts, registry=(_entry("image-pull-access"),))[0]
        self.assertEqual(g.level, "WARN")

    def test_requires_scopes_to_pinned_images(self):
        self.assertTrue(cr._req_image_pull_access(self._recipe()))
        self.assertFalse(cr._req_image_pull_access({"envelope": {}, "serving": {}}))  # no images → no-op


def _entry(cap_id):
    return next(e for e in cr.REGISTRY if e.id == cap_id)


def _os_environ():
    import os

    return dict(os.environ)


if __name__ == "__main__":
    unittest.main(verbosity=2)
