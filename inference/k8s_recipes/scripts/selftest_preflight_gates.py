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

"""selftest_preflight_gates.py — the cross-file wiring of the three pre-run preflight gates.

The registry's OWN pure surface (paths/verdict/requires/gap/image-inspect) is covered in
selftest_capability_registry.py. This file covers the two seams that live OUTSIDE capability_registry:

  Gate 1 (model-cache-integrity): preflight.parse_cache_integrity_report — the busybox KEY=val report the
          read-only PVC-mount probe emits → the facts dict capability_registry.model_cache_verdict consumes.
          Fault fixture → ABSENT/FAIL · healthy → AVAILABLE/PASS · probe-error → UNKNOWN.
  Gate 2 (sweep ≤ served max-num-seqs): check_invariants.resolve_max_num_seqs (the cap precedence, incl. the
          explicit vLLM 256 default) + check_cell severity — FAIL for llm-perf, WARN (cannot-assert) for
          SGLang, silent when the recipe doesn't sweep.

Pure/offline (no cluster). Run `python3 scripts/selftest_preflight_gates.py` or via `make test`. Exit 0 = pass.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


pf = _load("preflight")
cr = _load("capability_registry")
ci = _load("check_invariants")
mc = _load("model_cache")


# ── Gate 1: preflight.parse_cache_integrity_report → model_cache_verdict ─────────────────────────────
# THE SIZE LINES ARE PART OF A REAL REPORT, so the fixtures carry them. cache_probe_script always emits
# SHARD_BYTES/SHARD_MIN/SHARD_MAX/SHARD_N, and a report where SHARDS>0 but nothing could be SIZED means the
# sizing half of the probe returned nothing — which used to switch every byte guard off and fall through to
# COMPLETE. That is now UNKNOWN (see test_sized_nothing_is_unknown below), so a fixture that omits the size
# lines is describing a report the probe cannot produce.
def _sized(report: str, n: int, per: int = 4_000_000_000) -> str:
    """Append the per-file size block a real probe emits for `n` shards."""
    if n <= 0:
        return report + "SHARD_BYTES=\nSHARD_MIN=\nSHARD_MAX=\nSHARD_N=\n"
    return report + f"SHARD_BYTES={per * n}\nSHARD_MIN={per}\nSHARD_MAX={per}\nSHARD_N={n}\n"


_HEALTHY = _sized(
    "EXISTS=1\nCONFIG=1\nINDEX=1\nSHARDS=6\nREQ_SHARDS=6\nMISSING_SHARDS=0\n" "REF=183968f8\nREFSNAP=1\n",
    6,
)
_INCOMPLETE = _sized("EXISTS=1\nCONFIG=1\nINDEX=1\nSHARDS=0\nREF=183968f8\nREFSNAP=1\n", 0)  # partial shards
# Gap 1: config+index+SOME shards present (SHARDS>=1) but the index's weight_map references shards that
# DON'T resolve (MISSING_SHARDS>0) — a partial download the old ≥1-shard heuristic would falsely PASS.
_PARTIAL_SHARDS = _sized(
    "EXISTS=1\nCONFIG=1\nINDEX=1\nSHARDS=3\nREQ_SHARDS=4\nMISSING_SHARDS=1\n" "REF=183968f8\nREFSNAP=1\n",
    3,
)
_MISSING_DIR = "EXISTS=0\nCONFIG=0\nINDEX=0\nSHARDS=0\nREF=\nREFSNAP=0\n"
_DANGLING_REF = _sized(
    "EXISTS=1\nCONFIG=1\nINDEX=1\nSHARDS=6\nREQ_SHARDS=6\nMISSING_SHARDS=0\n" "REF=deadbeef\nREFSNAP=0\n",
    6,
)  # refs/main → absent snap


class TestGate1CacheReport(unittest.TestCase):
    def test_healthy_report_is_available(self):
        facts = pf.parse_cache_integrity_report(_HEALTHY, server_path="s/snapshots/183968f8")
        self.assertTrue(facts["exists"] and facts["config_json"] and facts["index_json"])
        self.assertEqual(facts["shard_count"], 6)
        self.assertTrue(facts["refs_consistent"])
        self.assertEqual(cr.model_cache_verdict(facts), cr.AVAILABLE)

    def test_incomplete_no_shards_is_absent(self):
        facts = pf.parse_cache_integrity_report(_INCOMPLETE)
        self.assertEqual(facts["shard_count"], 0)
        self.assertEqual(cr.model_cache_verdict(facts), cr.ABSENT)  # the crash-loop signature → block

    def test_missing_dir_is_absent(self):
        self.assertEqual(
            cr.model_cache_verdict(pf.parse_cache_integrity_report(_MISSING_DIR)),
            cr.ABSENT,
        )

    def test_partial_download_referenced_shard_missing_is_absent(self):
        # Gap 1: SHARDS=3 (≥1) would falsely PASS the old heuristic, but 1 of 4 index-referenced shards
        # dangles → the server still crash-loops → ABSENT/FAIL.
        facts = pf.parse_cache_integrity_report(_PARTIAL_SHARDS)
        self.assertEqual(facts["required_shards"], 4)
        self.assertEqual(facts["missing_shards"], 1)
        self.assertEqual(cr.model_cache_verdict(facts), cr.ABSENT)

    def test_single_file_model_no_index_is_available(self):
        # A single-file model (Qwen3-0.6B): config.json + ONE model.safetensors, NO index.json (INDEX=0,
        # REQ_SHARDS=0) → the loader resolves it fine → COMPLETE. The index is only required for multi-shard.
        facts = pf.parse_cache_integrity_report(
            _sized(
                "EXISTS=1\nCONFIG=1\nINDEX=0\nSHARDS=1\nREQ_SHARDS=0\nMISSING_SHARDS=0\nREF=c1899de2\nREFSNAP=1\n",
                1,
            )
        )
        self.assertFalse(facts["index_json"])
        self.assertEqual(facts["shard_count"], 1)
        self.assertEqual(cr.model_cache_verdict(facts), cr.AVAILABLE)

    def test_index_present_but_unenumerated_is_not_a_single_file_model(self):
        """Incomplete indexed snapshot. INDEX=1 with REQ_SHARDS=0 means the index file EXISTS
        and the probe enumerated nothing from it — not "this model has no index". The verdict used to be
        ('complete', 'config.json + 113 resolved shard(s), no shard index (single-file model)'): a proof
        string that is false about a 113-shard model whose shard set was never checked. install then wrote
        a completion sentinel on it. The old downloader also short-circuited on that file, so one probe
        hiccup survived re-installs and poisoned install, preflight and fleet while the server crash-looped
        on a missing shard. The current downloader always resumes/verifies before rewriting the sentinel.
        """
        facts = pf.parse_cache_integrity_report(
            _sized(
                "EXISTS=1\nCONFIG=1\nINDEX=1\nSHARDS=113\nREQ_SHARDS=0\nMISSING_SHARDS=0\nREF=\nREFSNAP=0\n",
                113,
            )
        )
        state, why = mc.cache_completeness(facts)
        self.assertEqual(state, mc.STATE_PRESENT_UNVERIFIED, why)
        self.assertNotIn("single-file", why)
        self.assertIn("IS PRESENT", why)  # names WHICH of the two situations it is
        self.assertEqual(cr.model_cache_verdict(facts), cr.ABSENT)  # a run must not start on this
        self.assertFalse(mc.sentinel_worthy(facts)[0])  # and it must never become permanent

    def test_sized_nothing_is_unknown(self):
        """SHARDS=113 with SHARD_N unmeasured is the sizing half of the probe returning nothing. It is not
        "the shards are empty" — every byte guard is simply inoperative, so the honest verdict is UNKNOWN.
        Reproduced against a real hub-layout fixture in busybox 1.36.1 by making `stat` unavailable: the
        probe exits 0 and emits SHARDS=10 with no sizes (the awk END block ran on empty input).
        """
        for _n_line in (
            "SHARD_BYTES=\nSHARD_MIN=\nSHARD_MAX=\nSHARD_N=\n",  # after the awk fix
            "SHARD_BYTES=0\nSHARD_MIN=0\nSHARD_MAX=0\nSHARD_N=0\n",
        ):  # the old zero shape
            facts = pf.parse_cache_integrity_report(
                "EXISTS=1\nCONFIG=1\nINDEX=1\nSHARDS=113\nREQ_SHARDS=113\nMISSING_SHARDS=0\n"
                "INDEX_TOTAL_BYTES=352284061280\nREF=\nREFSNAP=0\n" + _n_line
            )
            state, why = mc.cache_completeness(facts)
            self.assertEqual(state, mc.STATE_UNKNOWN, f"{_n_line!r} -> {state}: {why}")
            self.assertEqual(cr.model_cache_verdict(facts), cr.UNKNOWN)
            self.assertFalse(mc.sentinel_worthy(facts)[0])

    def test_dangling_ref_is_still_available(self):
        # Gap 3: refs/main is orthogonal to the pinned snapshot's completeness — a GC'd refs/main must NOT
        # fail a complete pinned cache. refs_consistent is parsed but informational only.
        facts = pf.parse_cache_integrity_report(_DANGLING_REF)
        self.assertFalse(facts["refs_consistent"])  # informational only now
        self.assertEqual(cr.model_cache_verdict(facts), cr.AVAILABLE)

    def test_empty_ref_is_consistent(self):
        # a snapshot with no refs/main is not "inconsistent" — nothing to be inconsistent with
        facts = pf.parse_cache_integrity_report(
            _sized(
                "EXISTS=1\nCONFIG=1\nINDEX=1\nSHARDS=3\nREQ_SHARDS=3\nMISSING_SHARDS=0\nREF=\nREFSNAP=0\n",
                3,
            )
        )
        self.assertTrue(facts["refs_consistent"])
        self.assertEqual(cr.model_cache_verdict(facts), cr.AVAILABLE)

    def test_probe_error_dict_is_unknown(self):
        # what probe_model_cache_integrity returns when the mounter pod never readies
        self.assertEqual(
            cr.model_cache_verdict({"probe_error": "mounter pod did not ready"}),
            cr.UNKNOWN,
        )


# ── Gap 2: nvlink-p2p is init-time cached — a tp>1 cell must RE-PROBE live in preflight ───────────────
class TestGate2P2pLiveReprobe(unittest.TestCase):
    """The stale-fact fix: preflight actively re-probes NVLink P2P for a tp>1 recipe rather than trusting
    only the init-time-cached profile fact. The trigger + merge decisions are PURE and tested here.
    """

    def test_agg_tp_gt1_perf_cell_requires_and_triggers_live_probe(self):
        recipe = {
            "envelope": {"gpu_type": "GB300", "arch": "arm64", "scenario": "llm-perf"},
            "serving": {"tp": 8},
        }  # aggregated tp>1 → engages P2P
        required = any(e.id == "nvlink-p2p" for e in cr.required_capabilities(recipe))
        self.assertTrue(required)
        self.assertTrue(pf.should_probe_p2p_live(required, cached_fresh=False))  # must probe now

    def test_fresh_result_in_hand_skips_double_probe(self):
        self.assertFalse(pf.should_probe_p2p_live(required=True, cached_fresh=True))

    def test_tp1_cell_never_probes(self):
        recipe = {
            "envelope": {"gpu_type": "GB300", "arch": "arm64", "scenario": "llm-perf"},
            "serving": {"tp": 1},
        }
        required = any(e.id == "nvlink-p2p" for e in cr.required_capabilities(recipe))
        self.assertFalse(required)
        self.assertFalse(pf.should_probe_p2p_live(required, cached_fresh=False))

    def test_live_disabled_overrides_stale_cached_healthy(self):
        # A live probe must override a stale healthy result from profile initialization.
        merged = pf.merge_p2p_fact({"state": "healthy"}, {"state": "disabled", "route_unhealthy": True})
        self.assertEqual(merged["state"], "disabled")
        self.assertIs(merged["route_unhealthy"], True)

    def test_live_unknown_keeps_cached_fact(self):
        # probe couldn't schedule → safe-degrade: keep the measured cached fact, never a false block
        merged = pf.merge_p2p_fact({"state": "healthy"}, {"state": "unknown"})
        self.assertEqual(merged["state"], "healthy")

    def test_live_probe_error_none_keeps_cached_fact(self):
        merged = pf.merge_p2p_fact({"state": "disabled"}, None)
        self.assertEqual(merged["state"], "disabled")


# ── Gate 2: check_invariants.resolve_max_num_seqs + check_cell severity ──────────────────────────────
class TestGate2Resolve(unittest.TestCase):
    def test_explicit_field_wins(self):
        self.assertEqual(ci.resolve_max_num_seqs({"stack": "vllm-agg", "max_num_seqs": 300})[0], 300)

    def test_vllm_max_num_seqs_field(self):
        cap, src = ci.resolve_max_num_seqs({"stack": "vllm-agg", "vllm_max_num_seqs": 128})
        self.assertEqual(cap, 128)
        self.assertIn("vllm_max_num_seqs", src)

    def test_extra_args_arg(self):
        cap, _ = ci.resolve_max_num_seqs({"stack": "vllm-agg", "extra_args": ["--max-num-seqs 512"]})
        self.assertEqual(cap, 512)

    def test_vllm_default_256_when_unset(self):
        cap, src = ci.resolve_max_num_seqs({"stack": "vllm-agg"})
        self.assertEqual(cap, 256)
        self.assertIn("256", src)

    def test_sglang_unset_is_unknown(self):
        cap, src = ci.resolve_max_num_seqs({"stack": "sglang-disagg"})
        self.assertIsNone(cap)  # don't assume a cap → WARN, not FAIL
        self.assertEqual(src, "unknown")

    def test_disagg_role_arg(self):
        sv = {
            "stack": "vllm-disagg",
            "disagg": {"decode": {"extra_args": ["--max-num-seqs 200"]}},
        }
        self.assertEqual(ci.resolve_max_num_seqs(sv)[0], 200)


# ── Gate 3: preflight.probe_image_pull_access — the pre-run CREDENTIAL gate (seam over krun + http) ───
class TestGate3PullAccessWrapper(unittest.TestCase):
    """The preflight seam OUTSIDE capability_registry: read the namespace IMAGE_PULL_SECRET dockerconfig via
    krun, hand it to cr.probe_image_pull_access. The registry auth probe itself is covered in
    selftest_capability_registry; here we assert the secret-read → UNKNOWN degrade + the end-to-end FAIL.
    """

    _IMAGE = "nvcr.io/nvidia/llmb-server@sha256:aabbcc"
    _RECIPE = {
        "envelope": {"provenance": {"image_digest": "nvcr.io/nvidia/llmb-server@sha256:aabbcc"}},
        "serving": {},
    }
    _PROF = {"IMAGE_PULL_SECRET": "ngc-registry"}

    def setUp(self):
        self._krun = pf.krun
        self._http = cr._default_pull_http

    def tearDown(self):
        pf.krun = self._krun
        cr._default_pull_http = self._http

    def _dockercfg_b64(self):
        import base64 as b
        import json as j

        auth = b.b64encode(b"$oauthtoken:NGCKEY").decode()
        return b.b64encode(j.dumps({"auths": {"nvcr.io": {"auth": auth}}}).encode()).decode()

    def test_unreadable_secret_degrades_to_unknown(self):
        pf.krun = lambda args, timeout=30: (
            1,
            "",
            "Error from server (Forbidden)",
        )  # RBAC-denied read
        out = pf.probe_image_pull_access("ns", self._PROF, self._RECIPE)
        self.assertIsNone(out["results"])  # → UNKNOWN, not FAIL
        self.assertEqual(cr.image_pull_access_verdict(out["results"]), cr.UNKNOWN)

    def test_unparseable_secret_degrades_to_unknown(self):
        pf.krun = lambda args, timeout=30: (0, "%%%not-base64%%%", "")
        out = pf.probe_image_pull_access("ns", self._PROF, self._RECIPE)
        self.assertIsNone(out["results"])

    def test_forbidden_end_to_end_is_fail(self):
        cfg = self._dockercfg_b64()
        pf.krun = lambda args, timeout=30: (0, cfg, "")
        # canned registry: the cred authenticates but the image repo scope is 403
        from urllib.parse import parse_qs, urlparse

        def http(method, url, headers=None):
            headers = headers or {}
            if "/manifests/" in url and not headers.get("Authorization", "").startswith("Bearer"):
                return (
                    401,
                    {"WWW-Authenticate": 'Bearer realm="https://nvcr.io/proxy_auth",service="nvcr.io"'},
                    "",
                )
            if "proxy_auth" in url:
                scope = parse_qs(urlparse(url).query).get("scope", [""])[0]
                return (403, {}, "") if scope == "repository:nvidia/llmb-server:pull" else (200, {}, '{"token":"T"}')
            return 200, {}, ""

        cr._default_pull_http = http
        out = pf.probe_image_pull_access("ns", self._PROF, self._RECIPE)
        self.assertEqual(cr.image_pull_access_verdict(out["results"]), cr.ABSENT)
        g = cr.evaluate(
            self._RECIPE,
            {"image_pull_access": out},
            registry=tuple(e for e in cr.REGISTRY if e.id == "image-pull-access"),
        )[0]
        self.assertEqual(g.level, "FAIL")
        self.assertIn("ngc-registry", g.fix)


def _cell(tmp, recipe_yaml):
    d = Path(tmp)
    (d / "recipe.yaml").write_text(recipe_yaml)
    return d


class TestGate2CheckCell(unittest.TestCase):
    """Severity by scenario, asserted on the specific sweep-cap message (other unrelated findings ignored)."""

    def _findings(self, recipe_yaml):
        with tempfile.TemporaryDirectory() as td:
            return ci.check_cell(_cell(td, recipe_yaml))

    def test_llmperf_vllm_oversweep_fails(self):
        # no --max-num-seqs → vLLM default 256; sweep reaches 512 > 256 → INVALID measurement → FAIL
        y = (
            "envelope: {scenario: llm-perf, name: x, status: wip, arch: amd64, distribution: synthetic}\n"
            "serving: {stack: vllm-agg, model_repo: a/b, model_revision: r}\n"
            "bench: {sweep_concurrency: [128, 512], synthetic: {}}\n"
        )
        problems, warns = self._findings(y)
        self.assertTrue(any("served max-num-seqs is 256" in p for p in problems), problems)
        self.assertFalse(any("served max-num-seqs is 256" in w for w in warns))

    def test_llmperf_vllm_undersweep_ok(self):
        y = (
            "envelope: {scenario: llm-perf, name: x, status: wip, arch: amd64, distribution: synthetic}\n"
            "serving: {stack: vllm-agg, model_repo: a/b, model_revision: r, extra_args: ['--max-num-seqs 512']}\n"
            "bench: {sweep_concurrency: [128, 256], synthetic: {}}\n"
        )
        problems, warns = self._findings(y)
        self.assertFalse(any("only queues internally" in p for p in problems), problems)

    def test_sglang_oversweep_warns_cannot_assert(self):
        y = (
            "envelope: {scenario: llm-perf, name: x, status: wip, arch: amd64, distribution: synthetic}\n"
            "serving: {stack: sglang-disagg, model_repo: a/b, model_revision: r}\n"
            "bench: {sweep_concurrency: [128, 512], synthetic: {}}\n"
        )
        problems, warns = self._findings(y)
        self.assertTrue(
            any("cannot determine the served concurrency cap" in w for w in warns),
            warns,
        )
        self.assertFalse(any("only queues internally" in p for p in problems), problems)

    def test_no_sweep_is_silent(self):
        y = (
            "envelope: {scenario: llm-perf, name: x, status: wip, arch: amd64, distribution: synthetic}\n"
            "serving: {stack: vllm-agg, model_repo: a/b, model_revision: r}\n"
            "bench: {synthetic: {}}\n"
        )
        problems, warns = self._findings(y)
        self.assertFalse(any("max-num-seqs" in m for m in problems + warns), (problems, warns))


class TestWholeNodeCoscheduling(unittest.TestCase):
    """Validate combined server and colocated-benchmark resource demand."""

    # Representative allocatable and free capacity after system workloads.
    ALLOC_CPU, ALLOC_MEM, FREE_CPU, FREE_MEM = 192000, 2015, 189700, 2000
    BENCH = {
        "manifest": "bench-job.yaml",
        "cpu_m": 16000,
        "mem_gib": 16,
        "cpu_raw": "16",
        "mem_raw": "16Gi",
    }

    _DEFAULT = object()  # distinct sentinel: None is a MEANINGFUL value here (free capacity UNKNOWN)

    def _verdict(self, wn_cpu, bench=_DEFAULT, free_cpu=_DEFAULT, free_mem=_DEFAULT):
        return pf._coschedule_verdict(
            str(wn_cpu),
            "1800Gi",
            wn_cpu * 1000,
            1800,
            self.BENCH if bench is self._DEFAULT else bench,
            self.ALLOC_CPU,
            self.ALLOC_MEM,
            self.FREE_CPU if free_cpu is self._DEFAULT else free_cpu,
            self.FREE_MEM if free_mem is self._DEFAULT else free_mem,
            "zfprj",
            "example-gpu-cluster",
        )

    def test_combined_cpu_demand_fails(self):
        v = self._verdict(180)
        self.assertTrue(v, "180 + 16 on a 192-cpu node must FAIL")
        msg, fix = v
        # Include both requests and their combined demand in the diagnostic.
        self.assertIn("WHOLE_NODE_CPU=180", msg)
        self.assertIn("BENCH_CPU_REQUEST=16", msg)
        self.assertIn("196", msg)
        self.assertIn("podAffinity", msg)
        self.assertIn("WHOLE_NODE_CPU", fix)
        self.assertIn("BENCH_CPU_REQUEST", fix)

    def test_recommendation_comes_from_the_tighter_ceiling(self):
        """free <= allocatable, and free is what the scheduler compares against. Recommending 176 (from
        allocatable) would still not fit once DaemonSets have taken ~2.3 cpu."""
        _, fix = self._verdict(180)
        self.assertIn("≤173", fix)
        self.assertNotIn("≤176", fix)

    def test_falls_back_to_allocatable_when_free_is_unknown(self):
        v = self._verdict(180, free_cpu=None, free_mem=None)
        self.assertTrue(v)
        self.assertIn("≤176", v[1])

    def test_the_operator_fix_passes(self):
        self.assertFalse(self._verdict(168), "168 + 16 fits in ~189.7 free — must PASS")

    def test_silent_when_bench_can_schedule_anywhere(self):
        """Must not fire for cells whose bench has no required same-host affinity — i.e. all but one."""
        self.assertFalse(self._verdict(180, bench={}))

    def test_memory_budget_too(self):
        v = pf._coschedule_verdict(
            "100",
            "2010Gi",
            100000,
            2010,
            self.BENCH,
            self.ALLOC_CPU,
            self.ALLOC_MEM,
            self.FREE_CPU,
            self.FREE_MEM,
            "n",
            "p",
        )
        self.assertTrue(v)
        self.assertIn("WHOLE_NODE_MEM=2010Gi", v[0])

    def test_unknown_bench_request_is_not_zero(self):
        """An unparseable request must never read as 'the bench pod needs nothing'."""
        self.assertFalse(
            self._verdict(
                180,
                bench={"cpu_m": None, "mem_gib": None, "cpu_raw": "?", "mem_raw": "?"},
            )
        )


class TestBenchCoscheduleDemand(unittest.TestCase):
    """Reads the RENDERED manifests — what will actually be applied."""

    ROOT = Path(__file__).resolve().parent.parent
    KVBM = "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"
    GLM = "recipes/llm-perf/Glm5/B200_k8s/Agg/1k_1k/glm5-fp8-b200-sglang15-agg-c4-1024"
    PROF = {
        "BENCH_CPU_REQUEST": "16",
        "RUN_ID": "r",
        "NAMESPACE": "n",
        "BENCH_NODE_SELECTOR": "",
        "IMAGE_PULL_SECRET": "s",
    }

    def test_colocated_cell_reports_its_demand(self):
        d = pf.bench_coschedule_demand(self.ROOT / self.KVBM, self.PROF)
        self.assertEqual(d.get("cpu_m"), 16000)
        self.assertEqual(d.get("mem_gib"), 16)  # a LITERAL in the manifest, not a profile var

    def test_non_colocated_cell_reports_nothing(self):
        self.assertEqual(pf.bench_coschedule_demand(self.ROOT / self.GLM, self.PROF), {})

    def test_multi_document_manifest_is_parsed(self):
        """The bench manifest bundles ServiceAccount/Role/RoleBinding with the Job; safe_load() alone
        returns only the first document and the check would silently no-op."""
        self.assertTrue(pf.bench_coschedule_demand(self.ROOT / self.KVBM, self.PROF))

    def test_unsubstituted_vars_do_not_break_the_parse(self):
        """`{ claimName: ${MODEL_CACHE_PVC} }` — the `{` of `${...}` opens a nested flow mapping and the
        YAML parse dies. With a bare except that turned the whole gate into a no-op."""
        d = pf.bench_coschedule_demand(self.ROOT / self.KVBM, {})
        self.assertNotIn("parse_error", d)
        self.assertEqual(d.get("mem_gib"), 16)

    def test_absent_bench_cpu_request_uses_the_launchers_default(self):
        """BENCH_CPU_REQUEST IS OPTIONAL. It is not in CRITICAL_PROFILE_VARS, and sweep.sh/submit.sh/
        dryrun.sh each default it to 16 before envsubst — so a profile that omits it still runs a 16-cpu
        bench pod. preflight substituted from the PROFILE ALONE, so cpu_m came back None, the 'unreadable'
        WARN needed BOTH cpu and mem to be None (memory is a literal 16Gi, so it never fired), and
        _coschedule_verdict skipped the cpu comparison entirely — the exact 180+16 > 192 case this gate was
        written for, silently not checked, then rendered as `180+__llmb_unset__ cpu` inside a PASS.
        """
        d = pf.bench_coschedule_demand(self.ROOT / self.KVBM, {})
        self.assertEqual(d.get("cpu_m"), 16000)
        self.assertEqual(d.get("cpu_raw"), "16")
        self.assertNotIn(pf._UNSET_TOKEN, d.get("cpu_raw", ""))
        # The launcher default applies when BENCH_CPU_REQUEST is omitted.
        v = pf._coschedule_verdict(
            "180",
            "1500Gi",
            180000,
            1500,
            d,
            192000,
            2000,
            189700,
            1900,
            "node-a",
            "prof",
        )
        self.assertTrue(v, "180 + 16 > 192 must FAIL even when the profile omits BENCH_CPU_REQUEST")
        self.assertIn("BENCH_CPU_REQUEST=16", v[0])
        # An explicit profile value still wins over the default.
        self.assertEqual(
            pf.bench_coschedule_demand(self.ROOT / self.KVBM, {"BENCH_CPU_REQUEST": "64"}).get("cpu_m"),
            64000,
        )

    def test_the_default_matches_what_the_launchers_actually_do(self):
        """Derived from the shell, not asserted: if sweep.sh/submit.sh/dryrun.sh change their default, this
        fails rather than letting preflight quietly model a bench pod nobody launches.
        """
        for _s in ("sweep.sh", "submit.sh", "dryrun.sh"):
            txt = (self.ROOT / "scripts" / _s).read_text()
            m = re.search(r"\$\{BENCH_CPU_REQUEST:=([^}]*)\}", txt)
            self.assertIsNotNone(m, f"{_s} no longer defaults BENCH_CPU_REQUEST")
            self.assertEqual(
                m.group(1),
                pf._RUNTIME_DEFAULTS["BENCH_CPU_REQUEST"],
                f"{_s} defaults to {m.group(1)}, preflight models " f"{pf._RUNTIME_DEFAULTS['BENCH_CPU_REQUEST']}",
            )

    def test_a_var_with_no_launcher_default_still_reads_as_unknown(self):
        """The absence-as-signal half must survive the defaults table: a var nobody defaults stays
        UNPARSEABLE, so it reads as UNKNOWN rather than as zero."""
        out = pf._subst_profile_vars("cpu: ${SOME_VAR_NOBODY_SETS}", {})
        self.assertIn(pf._UNSET_TOKEN, out)
        self.assertIsNone(pf._cpu_millicores(pf._UNSET_TOKEN))

    def test_any_unreadable_request_is_reported_not_just_both(self):
        """The WARN guard was `cpu is None AND mem is None`. Memory is a literal, so it was unreachable."""
        self.assertEqual(pf.coschedule_unreadable({"cpu_m": None, "mem_gib": 16}), ["cpu"])
        self.assertEqual(pf.coschedule_unreadable({"cpu_m": 16000, "mem_gib": None}), ["memory"])
        self.assertEqual(
            pf.coschedule_unreadable({"cpu_m": None, "mem_gib": None}),
            ["cpu", "memory"],
        )
        self.assertEqual(pf.coschedule_unreadable({"cpu_m": 16000, "mem_gib": 16}), [])
        # and a half-read demand must never be judged as fitting
        self.assertFalse(
            pf._coschedule_verdict(
                "180",
                "1500Gi",
                180000,
                1500,
                {"cpu_m": None, "mem_gib": 16, "cpu_raw": "?", "mem_raw": "16Gi"},
                192000,
                2000,
                189700,
                1900,
                "n",
                "p",
            )
        )


class TestDisaggCacheAccessMode(unittest.TestCase):
    """Frontend metadata and both workers need one cache attachable across their nodes."""

    DISAGG = {
        "stack": "sglang-disagg",
        "disagg": {"prefill": {"tp": 8}, "decode": {"tp": 8}},
    }

    def test_rwx_and_rox_are_multi_node_safe(self):
        self.assertIsNone(pf.disagg_cache_access_issue(self.DISAGG, ["ReadWriteMany"]))
        self.assertIsNone(pf.disagg_cache_access_issue(self.DISAGG, ["ReadOnlyMany"]))

    def test_rwo_fails_with_actionable_reason(self):
        issue = pf.disagg_cache_access_issue(self.DISAGG, ["ReadWriteOnce"])
        self.assertIn("ReadWriteOnce", issue)
        self.assertIn("Multi-Attach", issue)

    def test_one_worker_plus_frontend_still_requires_shared_access(self):
        one_worker = {"stack": "future-disagg", "disagg": {"decode": {"tp": 8}}}
        self.assertIn("Multi-Attach", pf.disagg_cache_access_issue(one_worker, ["ReadWriteOnce"]))

    def test_aggregate_rwo_is_unchanged(self):
        self.assertIsNone(pf.disagg_cache_access_issue({"stack": "sglang-agg"}, ["ReadWriteOnce"]))


class TestLazyCacheResolution(unittest.TestCase):
    """A claim is REQUIRED where it is mounted, and only there — and it is still required there.

    The dangerous direction is quietly not requiring it where it IS mounted — so that is asserted first,
    end to end, by actually running main() and catching the refusal."""

    ROOT = Path(__file__).resolve().parent.parent
    KVBM = "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"

    def test_every_shipped_cell_still_requires_a_claim(self):
        cells = [p.parent for p in (self.ROOT / "recipes").glob("**/recipe.yaml") if (p.parent / "rendered").is_dir()]
        self.assertTrue(cells)
        missed = [c.name for c in cells if not pf.cell_mounts_model_cache(c)]
        self.assertEqual(missed, [], f"these cells mount the cache but would not be gated: {missed}")

    def test_main_still_refuses_a_mounting_cell_with_no_claim(self):
        """FAIL-CLOSED, end to end. This runs main() far enough to hit the gate; the refusal happens before
        any kubectl call, so it needs no cluster."""
        with tempfile.TemporaryDirectory() as td:
            envf = Path(td) / "nocache.env"
            envf.write_text('NAMESPACE="ns"\nGPU_PRODUCT="NVIDIA-B200"\nIMAGE_PULL_SECRET="s"\n')
            argv = sys.argv[:]
            sys.argv = [
                "preflight.py",
                str(self.ROOT / self.KVBM),
                str(td) + "/nocache",
                "--stage-only",
            ]
            try:
                with self.assertRaises(SystemExit) as ctx:
                    pf.main()
            finally:
                sys.argv = argv
            self.assertIn("no model-cache PVC resolved", str(ctx.exception))
            self.assertIn("MODEL_CACHE_PVC_QWEN3_0_6B", str(ctx.exception))


class TestWizardWholeNodeReserve(unittest.TestCase):
    """The number is BORN in wizard_init; an operator copying from `kubectl describe node` makes exactly
    the mistake that caused this."""

    def setUp(self):
        self.w = _load("wizard_init")

    def test_large_node_behaviour_is_unchanged(self):
        self.assertEqual(self.w.whole_node_cpu(192000), 163)  # 85% still wins on a big node
        self.assertLessEqual(163 + 16, 192)

    def test_small_node_is_capped_so_the_bench_still_fits(self):
        """85% of 32 cpu is 27, and 27+16=43 does not fit 32."""
        v = self.w.whole_node_cpu(32000)
        self.assertLessEqual(v + self.w.BENCH_RESERVE_CPU_CORES, 32)

    def test_undetected_stays_zero(self):
        self.assertEqual(self.w.whole_node_cpu(0), 0)
        self.assertEqual(self.w.whole_node_mem_gib(0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
