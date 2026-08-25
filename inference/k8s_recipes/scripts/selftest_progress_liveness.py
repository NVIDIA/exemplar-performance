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

"""Behavioral tests for the progress-aware liveness probe.

The tests run the shipped probe against representative metrics. A failed probe requires both a flat generated-token counter and active requests; missing, invalid, or reset metrics leave the pod healthy.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

K8S = Path(__file__).resolve().parent.parent
PROBE = K8S / "serving" / "_shared" / "progress_liveness.sh"

DYN = (
    'dynamo_frontend_output_tokens_total{model="qwen3-0-6b"} %s\n'
    'dynamo_frontend_inflight_requests{model="qwen3-0-6b"} %s\n'
)
VLLM = 'vllm:generation_tokens_total{model="m"} %s\n' 'vllm:num_requests_running{model="m"} %s\n'


class ProbeTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.state = self.root / "state"
        self.addCleanup(self.td.cleanup)

    def _curl(self, body: str | None, *, rc: int = 0):
        c = self.bin / "curl"
        if body is None:
            c.write_text("#!/bin/sh\nexit 7\n")
        else:
            c.write_text("#!/bin/sh\nprintf '%s' " + "'" + body.replace("'", "'\\''") + f"'\nexit {rc}\n")
        c.chmod(0o755)

    def _probe(self, body: str | None, *, rc: int = 0) -> int:
        self._curl(body, rc=rc)
        env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "PROGRESS_PROBE_STATE": str(self.state),
            "SERVER_URL": "http://fake:8000",
        }
        return subprocess.run(["sh", str(PROBE)], capture_output=True, text=True, env=env).returncode

    # ---- stalled-generation detection ---------------------------------------
    def test_flat_tokens_with_work_outstanding_is_UNHEALTHY(self):
        """Active requests with an unchanged token counter indicate a stall."""
        self.assertEqual(self._probe(DYN % ("1102016", "32")), 0, "first sample can never fail")
        self.assertEqual(self._probe(DYN % ("1102016", "32")), 1, "flat + 32 in flight = wedged")

    def test_it_takes_a_REPEAT_to_fail(self):
        """One observation is never enough — kubelet's failureThreshold owns the patience."""
        self.assertEqual(self._probe(DYN % ("500", "8")), 0)

    def test_first_probe_with_ZERO_tokens_and_work_in_flight_is_healthy(self):
        """The first probe establishes a baseline, even when requests are already active."""
        self.assertEqual(self._probe(DYN % ("0", "4")), 0)
        # A second unchanged sample with active work detects the stall.
        self.assertEqual(self._probe(DYN % ("0", "4")), 1)

    # ---- states that must remain healthy ------------------------------------
    def test_progressing_server_is_healthy(self):
        self._probe(DYN % ("1000", "16"))
        self.assertEqual(self._probe(DYN % ("2000", "16")), 0)

    def test_idle_between_rungs_is_healthy_however_long(self):
        """Tokens flat for hours between rungs — but nothing in flight, so not a stall."""
        self._probe(DYN % ("5000", "0"))
        for _ in range(50):
            self.assertEqual(self._probe(DYN % ("5000", "0")), 0)

    def test_model_load_is_healthy(self):
        """Cold load: zero tokens ever generated, nothing admitted yet."""
        self._probe(DYN % ("0", "0"))
        self.assertEqual(self._probe(DYN % ("0", "0")), 0)

    def test_unreachable_scrape_is_healthy(self):
        self._probe(DYN % ("100", "8"))
        self.assertEqual(self._probe(None), 0, "cannot measure != wedged")

    def test_truncated_body_is_healthy(self):
        self._probe(DYN % ("100", "8"))
        self.assertEqual(self._probe(DYN % ("100", "8"), rc=18), 0)

    def test_http200_without_counters_is_healthy(self):
        """Frontend restarted, series not yet registered — UNKNOWN, not a stall."""
        self._probe(DYN % ("100", "8"))
        self.assertEqual(self._probe("# HELP something else\nunrelated_metric 1\n"), 0)

    def test_counter_regression_is_healthy_and_resets(self):
        """A restart drops the counter. That is not a stall, and must not leave a live strike."""
        self._probe(DYN % ("9999", "8"))
        self.assertEqual(self._probe(DYN % ("5", "8")), 0)
        self.assertEqual(
            self._probe(DYN % ("5", "8")),
            1,
            "after the reset, a real stall still trips",
        )

    def test_unknown_runtime_is_healthy(self):
        self.assertEqual(self._probe("mystery_engine_tokens 5\nmystery_inflight 3\n"), 0)
        self.assertEqual(self._probe("mystery_engine_tokens 5\nmystery_inflight 3\n"), 0)

    def _probe_out(self, body):
        self._curl(body)
        env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "PROGRESS_PROBE_STATE": str(self.state),
            "SERVER_URL": "http://fake:8000",
        }
        p = subprocess.run(["sh", str(PROBE)], capture_output=True, text=True, env=env)
        return p.returncode, p.stderr

    def test_unknown_runtime_WARNS_that_protection_is_inactive(self):
        """An unsupported runtime stays healthy but reports that stall detection is inactive."""
        rc, err = self._probe_out("mystery_engine_tokens 5\nmystery_inflight 3\n")
        self.assertEqual(rc, 0, "unsupported runtime must never kill the pod")
        self.assertIn("Stall detection is unavailable", err)

    def test_the_unsupported_warning_is_emitted_ONCE_not_every_poll(self):
        """The unsupported-runtime warning is emitted once per container."""
        body = "mystery_engine_tokens 5\nmystery_inflight 3\n"
        _, first = self._probe_out(body)
        _, second = self._probe_out(body)
        self.assertIn("Stall detection is unavailable", first)
        self.assertNotIn("Stall detection is unavailable", second)

    def test_a_transient_scrape_failure_does_NOT_claim_unsupported_runtime(self):
        """A transient scrape failure is distinct from an unsupported runtime."""
        _, err = self._probe_out(None)
        self.assertNotIn("Stall detection is unavailable", err)

    def test_an_unmeasurable_window_clears_a_pending_strike(self):
        """A strike must not survive a blind gap and combine with a later one."""
        self._probe(DYN % ("100", "8"))
        self._probe(DYN % ("100", "8"))  # strike 1
        self.assertEqual(self._probe(None), 0)  # blind -> forget
        self.assertEqual(self._probe(DYN % ("100", "8")), 0, "history was cleared; start over")

    # ---- runtime-agnostic + real captured surface -------------------------
    def test_vllm_names_work_too(self):
        self.assertEqual(self._probe(VLLM % ("77", "4")), 0)
        self.assertEqual(self._probe(VLLM % ("77", "4")), 1)

    def test_per_worker_reexport_does_not_double_count_or_regress(self):
        """Frontend + per-worker families must not be summed; one resetting must not lower the read."""
        both = (DYN % ("198084", "15")) + 'dynamo_component_output_tokens_total{worker="1"} 198084\n'
        self.assertEqual(self._probe(both), 0)
        dropped = (DYN % ("198084", "15")) + 'dynamo_component_output_tokens_total{worker="1"} 0\n'
        self.assertEqual(self._probe(dropped), 1, "frontend family is unchanged -> still a stall")

    def test_real_captured_surface_parses_and_is_healthy(self):
        """The genuine /metrics from a live server — a real fixture, not a hand-written one."""
        real = K8S / "results" / "ttj86nq" / "server_metrics_surface.prom"
        if not real.is_file():
            self.skipTest("captured surface not present")
            return
        body = real.read_text()
        self.assertEqual(self._probe(body), 0)
        self.assertEqual(self._probe(body), 0, "an idle captured surface must never read as wedged")

    # ---- the delivery path: envsubst must not rewrite the script ----------
    def test_envsubst_at_apply_time_leaves_the_embedded_script_BYTE_IDENTICAL(self):
        """The probe is delivered inside a ConfigMap, and submit.sh derives its envsubst whitelist FROM
        THE MANIFEST ITSELF (`grep -oE '\\$\\{[A-Z_][A-Z0-9_]*\\}'`) — so a bare `${FOO}` anywhere in the
        embedded script joins that whitelist and is substituted to '' at apply time, because these are
        the probe's own shell locals and nothing exports them. `${FOO:-default}` survives (the `:-`
        breaks the pattern), so the damage is PARTIAL and passes a casual read: the first shipped
        version lost `${STATE}.unsupported` -> `.unsupported` while every unbraced `$STATE` stayed put.

        This drives the REAL pipeline — the same whitelist derivation, the same envsubst, an
        intentionally hostile empty environment — and demands the script come out unchanged. It pins
        the property (script survives delivery), not the one variable that happened to break.
        """
        import shutil

        if not shutil.which("envsubst"):
            self.skipTest("envsubst not installed (gettext)")
            return
        cms = sorted(K8S.glob("recipes/**/rendered/liveness-configmap.yaml"))
        self.assertTrue(
            cms,
            "no rendered liveness ConfigMap committed — is the probe wired in at all?",
        )
        src = PROBE.read_text()
        for cm in cms:
            raw = cm.read_text()
            wl = sorted({m for m in __import__("re").findall(r"\$\{[A-Z_][A-Z0-9_]*\}", raw)})
            wl = " ".join("$" + v.strip("${}") for v in wl)
            out = subprocess.run(
                ["envsubst", wl],
                input=raw,
                capture_output=True,
                text=True,
                env={"PATH": os.environ["PATH"], "NAMESPACE": "ns"},
            ).stdout
            # The ConfigMap indents the script by 4 under the data key; undo that to compare.
            body = "\n".join(
                ln[4:] if ln.startswith("    ") else ln
                for ln in out.splitlines()
                if ln.startswith("    ") or not ln.strip()
            )
            for i, line in enumerate(src.splitlines(), 1):
                if line.strip() and line not in body:
                    self.fail(
                        f"{cm.relative_to(K8S)}: envsubst altered the embedded script at source "
                        f"line {i}:\n    {line}\n  A bare ${{VAR}} was eaten at apply time — use "
                        f"$VAR or pre-compute it (see the header note in progress_liveness.sh)."
                    )

    def test_histogram_and_created_series_do_not_swamp_the_counter(self):
        body = (
            DYN % ("100", "4") + "dynamo_frontend_output_tokens_created 1784427117.9\n"
            'dynamo_frontend_output_tokens_bucket{le="1.0"} 900000\n'
        )
        self.assertEqual(self._probe(body), 0)
        self.assertEqual(self._probe(body), 1, "still a stall: the real counter is flat at 100")


if __name__ == "__main__":
    unittest.main(verbosity=2)
