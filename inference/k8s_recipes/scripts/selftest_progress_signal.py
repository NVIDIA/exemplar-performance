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

"""Exercise the wrapper's token-progress and outstanding-request handling.

The tests run the shipped shell functions against controlled metrics responses. They verify that
cleanup requires both stale token progress and active requests, and that unreadable metrics remain unknown.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

INJECT = Path(__file__).resolve().parent / "resilient_inject.py"

VLLM = 'vllm:generation_tokens_total{model="m"} 4200\n'
DYNAMO = 'dynamo_frontend_output_tokens_total{model="qwen3-0-6b"} 198084\n'
SGLANG = 'sglang:generation_tokens_total{model="m"} 777\n'
NOISE = (
    "# HELP vllm:generation_tokens_total Number of generation tokens.\n"
    "# TYPE vllm:generation_tokens_total counter\n"
    'vllm:prompt_tokens_total{model="m"} 999999\n'
    'some_input_tokens_total{model="m"} 555\n'
)
# Real shapes lifted from committed results/*/server_metrics_surface.prom captures.
VLLM_GAUGES = (
    'vllm:num_requests_running{engine="0",model_name="m"} 15.0\n'
    'vllm:num_requests_waiting{engine="0",model_name="m"} 0.0\n'
    'vllm:num_requests_waiting_by_reason{engine="0",model_name="m",reason="capacity"} 7.0\n'
)
DYNAMO_GAUGES = (
    'dynamo_frontend_inflight_requests{model="m"} 15\n'
    'dynamo_frontend_queued_requests{model="m"} 0\n'
    "dynamo_request_plane_inflight_requests 15\n"
)
IDLE_GAUGES = 'dynamo_frontend_inflight_requests{model="m"} 0\n' 'dynamo_frontend_queued_requests{model="m"} 0\n'


def _extract(*names: str) -> str:
    """Pull shell functions verbatim out of the injected wrapper."""
    src = INJECT.read_text()
    out = []
    for name in names:
        m = re.search(rf"^{re.escape(name)}\(\) \{{\n(.*?)^\}}$", src, re.M | re.S)
        if m:
            out.append(f"{name}() {{\n{m.group(1)}}}\n")
            continue
        m = re.search(rf"^{re.escape(name)}\(\)\s*\{{(.*)\}}\s*$", src, re.M)  # one-liner form
        if not m:
            raise AssertionError(f"could not extract {name}() from {INJECT}")
        out.append(f"{name}() {{{m.group(1)}}}\n")
    return "".join(out)


PARSERS = "_parse_tokens", "_parse_gauge", "_parse_inflight", "_parse_queued"
EMIT = (*PARSERS, "_metrics_fetch", "_json_esc", "_num_or", "_emit_status")


class ProgressSignalTest(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.root / "state").mkdir()
        self.addCleanup(self.td.cleanup)

    def _fake_curl(self, body: str | None, *, rc: int = 0):
        """body=None → curl exits non-zero with no output (unreachable).
        rc!=0 with a body → a TRUNCATED response: curl printed some bytes, then failed.
        """
        curl = self.bin / "curl"
        if body is None:
            curl.write_text("#!/bin/sh\nexit 7\n")
        else:
            payload = body.replace("'", "'\\''")
            curl.write_text(f"#!/bin/sh\nprintf '%s' '{payload}'\nexit {rc}\n")
        curl.chmod(0o755)

    def _run(self, script: str) -> str:
        env = {
            **os.environ,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "SERVER_URL": "http://fake:8000",
        }
        p = subprocess.run(["sh", "-c", script], capture_output=True, text=True, env=env, cwd=self.root)
        return p.stdout.strip()

    def _parse(self, fn: str, body: str) -> str:
        """Feed a /metrics body straight into one parser; report its output AND its exit status."""
        blob = body.replace("'", "'\\''")
        return self._run(_extract(*PARSERS) + f"printf '%s' '{blob}' | {fn}; echo \"RC=$?\"")

    # ---- the token progress signal ----------------------------------------
    def test_counts_vllm_tokens(self):
        self.assertIn("4200", self._parse("_parse_tokens", NOISE + VLLM))

    def test_counts_dynamo_tokens(self):
        """Dynamo metrics must be included alongside vLLM metrics."""
        out = self._parse(
            "_parse_tokens",
            NOISE.replace("vllm:generation_tokens_total", "x:unused") + DYNAMO,
        )
        self.assertIn("198084", out)
        self.assertIn("RC=0", out)

    def test_counts_sglang_tokens(self):
        self.assertIn("777", self._parse("_parse_tokens", SGLANG))

    def test_ignores_prompt_and_input_counters(self):
        """Only OUTPUT/generation counters are progress; prompt tokens must not inflate it."""
        self.assertIn("4200", self._parse("_parse_tokens", NOISE + VLLM))

    def test_ignores_created_timestamp_and_histogram_series(self):
        """`vllm:generation_tokens_created` is a UNIX timestamp (~1.78e9) and the histogram buckets are
        request counts — folding either into progress would swamp the real counter."""
        body = (
            VLLM + 'vllm:generation_tokens_created{model="m"} 1.7844271177907763e+09\n'
            'vllm:request_generation_tokens_bucket{le="1.0",model="m"} 900000\n'
        )
        out = self._parse("_parse_tokens", body)
        self.assertIn("4200", out)

    def test_per_worker_series_do_not_double_count(self):
        """A Dynamo frontend re-exports the same tokens per worker. Summing ACROSS families double-counts
        (and a worker restart then makes the sum fall) — the reading must be the max family, not the sum.
        """
        body = (
            DYNAMO + 'dynamo_component_output_tokens_total{worker="1"} 120000\n'
            'dynamo_component_output_tokens_total{worker="2"} 78084\n'
        )
        out = self._parse("_parse_tokens", body)
        self.assertIn("198084", out)
        self.assertNotIn("396168", out, "frontend + per-worker families must not be summed together")

    def test_one_family_resetting_does_not_lower_the_reading(self):
        """Worker restart → its counter drops to ~0. The frontend aggregate is untouched, so the reported
        progress must not fall (a fall reads as a stall)."""
        before = self._parse(
            "_parse_tokens",
            DYNAMO + 'dynamo_component_output_tokens_total{w="1"} 198000\n',
        )
        after = self._parse("_parse_tokens", DYNAMO + 'dynamo_component_output_tokens_total{w="1"} 3\n')
        self.assertIn("198084", before)
        self.assertIn("198084", after)

    def test_multiple_labels_of_one_family_ARE_summed(self):
        """Within a single metric name, label sets are shards of one counter (vLLM engine=0/1)."""
        body = 'vllm:generation_tokens_total{engine="0"} 100\n' 'vllm:generation_tokens_total{engine="1"} 40\n'
        self.assertIn("140", self._parse("_parse_tokens", body))

    def test_absent_token_counter_is_UNKNOWN_not_zero(self):
        """HTTP 200 whose body carries no token counter at all (frontend restarted) must FAIL the parse,
        not report 0 — a 0 would freeze progress and, once the gate is armed, false-kill.
        """
        out = self._parse("_parse_tokens", NOISE)
        self.assertIn("RC=1", out)
        self.assertNotIn("0\n", out.replace("RC=1", ""))

    # ---- fetch failure modes ----------------------------------------------
    def _fetch_tokens(self, body, *, rc: int = 0) -> str:
        self._fake_curl(body, rc=rc)
        return self._run(_extract("_metrics_fetch", *PARSERS) + '_metrics_fetch | _parse_tokens; echo "RC=$?"')

    def test_unreachable_scrape_is_EMPTY_not_zero(self):
        """The whole safety property: a failed read must not look like 'nothing generated'."""
        out = self._fetch_tokens(None)
        self.assertIn("RC=1", out, "a failed scrape must return non-zero")
        self.assertNotIn("0", out.replace("RC=1", ""), "a failed scrape must not emit a 0 count")

    def test_truncated_response_is_rejected_not_partially_counted(self):
        """curl that emits bytes then FAILS must not yield a count — a partial scrape is UNKNOWN,
        and a low partial sum would look like 'progress went backwards / stalled'."""
        out = self._fetch_tokens(NOISE + VLLM, rc=18)  # 18 = CURLE_PARTIAL_FILE
        self.assertIn("RC=1", out, "a curl that exits non-zero must fail the scrape")
        self.assertNotIn("4200", out, "a truncated response must not produce a token count")

    # ---- the outstanding-work signal ---------------------------------------
    def test_inflight_read_from_vllm_gauges(self):
        self.assertIn("15", self._parse("_parse_inflight", VLLM_GAUGES))

    def test_inflight_read_from_dynamo_gauges(self):
        out = self._parse("_parse_inflight", DYNAMO_GAUGES)
        self.assertIn("15", out)
        self.assertNotIn(
            "30",
            out,
            "frontend + request-plane in-flight are the same requests, not two sets",
        )

    def test_queued_excludes_the_per_reason_breakdown(self):
        """`vllm:num_requests_waiting_by_reason` is a breakdown of the SAME queue. A prefix match would
        double-count it; the reading must be the exact-name gauge (0), not 7."""
        out = self._parse("_parse_queued", VLLM_GAUGES)
        self.assertIn("0", out)
        self.assertNotIn("7", out)

    def test_absent_gauges_are_UNKNOWN(self):
        """No known in-flight gauge on the surface → parse FAILS. Absence must never read as 'no work',
        because 'no work' is what disarms the halt."""
        self.assertIn("RC=1", self._parse("_parse_inflight", NOISE + VLLM))

    # ---- _emit_status: end-to-end status.json ------------------------------
    def _emit(
        self,
        body,
        *,
        prev="0",
        phase="waiting-server",
        reason="",
        curl_rc=0,
        progress_utc="2000-01-01T00:00:00Z",
        idle_utc="2000-01-01T00:00:00Z",
        runs: int = 1,
    ) -> tuple[dict, str]:
        self._fake_curl(body, rc=curl_rc)
        sd = self.root / "state"
        (sd / ".progress_counter").write_text(prev + "\n")
        (sd / ".progress_utc").write_text(progress_utc + "\n")
        (sd / ".idle_utc").write_text(idle_utc + "\n")
        (sd / ".state").write_text(f"running {phase} {reason}\n")
        script = (
            '_STATE_DIR="' + str(sd) + '"\n'
            # _SIGNAL is set at wrapper top level, not inside a function — the harness must mirror that or
            # _emit_status silently falls back to its "running waiting-server " default and the .state
            # fixture above is never actually read.
            '_SIGNAL="$_STATE_DIR/.state"\n'
            "_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }\n"
            '_set_state() { echo "$1 $2 $3" > "$_STATE_DIR/.state"; }\n'
            + _extract(*EMIT)
            + "_emit_status\n" * runs
            + 'cat "$_STATE_DIR/status.json"\n'
            'echo "PROGRESS_UTC=$(cat "$_STATE_DIR/.progress_utc")"\n'
            'echo "IDLE_UTC=$(cat "$_STATE_DIR/.idle_utc")"\n'
        )
        self._raw = self._run(script)
        first = self._raw.splitlines()[0] if self._raw else ""
        return json.loads(first), self._raw

    def _emit_raw(self, *a, **kw) -> str:
        """Like _emit but tolerant of INVALID json — so a broken status.json reports as a named FAIL with
        the offending bytes, not as an opaque decoder traceback."""
        try:
            self._emit(*a, **kw)
        except json.JSONDecodeError:
            pass
        return self._raw.splitlines()[0] if self._raw else ""

    def test_dynamo_tokens_advance_phase_to_generating(self):
        st, _ = self._emit(DYNAMO + DYNAMO_GAUGES)
        self.assertEqual(st["phase"], "generating")
        self.assertEqual(st["progress_counter"], 198084)

    def test_phase_stays_waiting_server_before_any_tokens(self):
        st, _ = self._emit('dynamo_frontend_output_tokens_total{model="m"} 0\n')
        self.assertEqual(st["phase"], "waiting-server")

    def test_progress_counter_is_a_monotonic_high_water_mark(self):
        """A counter that goes BACKWARDS (frontend restart) must not publish a lower progress_counter —
        a published regression is indistinguishable from a stall and, once armed, is a false kill.
        """
        st, out = self._emit(
            'dynamo_frontend_output_tokens_total{model="m"} 5\n',
            prev="198084",
            phase="generating",
        )
        self.assertEqual(
            st["progress_counter"],
            198084,
            "high-water mark must be retained, not overwritten",
        )
        self.assertIn(
            "PROGRESS_UTC=2000-01-01",
            out,
            "a counter reset is not progress — progress_utc must not be refreshed by it",
        )

    def test_high_water_survives_across_repeated_emits(self):
        st, _ = self._emit(
            'dynamo_frontend_output_tokens_total{model="m"} 7\n',
            prev="900",
            phase="generating",
            runs=3,
        )
        self.assertEqual(st["progress_counter"], 900)

    def test_unreachable_metrics_do_NOT_stale_progress(self):
        """Arming the stall gate is only safe if an unmeasurable window refreshes progress_utc."""
        st, out = self._emit(None, prev="500", phase="generating")
        self.assertNotIn(
            "PROGRESS_UTC=2000-01-01",
            out,
            "an unreachable scrape must not leave progress stale → false stall-kill",
        )
        self.assertEqual(
            st["progress_note"],
            "metrics-unreachable",
            "the cause must be visible, not silent",
        )
        self.assertEqual(st["reason"], "metrics-unreachable")
        self.assertEqual(st["progress_counter"], 500, "counter must hold, never reset to 0")

    def test_unreachable_metrics_do_not_fabricate_a_phase_advance(self):
        st, _ = self._emit(None, prev="0", phase="waiting-server")
        self.assertEqual(st["phase"], "waiting-server")

    def test_http200_without_token_counter_is_UNKNOWN_and_named(self):
        """A restarted frontend answers 200 with no counter series. That is UNMEASURABLE, and must both
        refresh progress_utc (no false stall) and say WHY."""
        st, out = self._emit(NOISE + DYNAMO_GAUGES, prev="500", phase="generating")
        self.assertEqual(st["progress_note"], "metrics-no-token-counter")
        self.assertNotIn("PROGRESS_UTC=2000-01-01", out)
        self.assertEqual(st["progress_counter"], 500)

    def test_inflight_published_and_idle_utc_HELD_while_work_outstanding(self):
        """Work outstanding → the outstanding-work window keeps accruing (idle_utc is NOT refreshed).
        This is the only path that can ever arm a halt."""
        st, out = self._emit(DYNAMO + DYNAMO_GAUGES, prev="198084", phase="generating")
        self.assertEqual(st["inflight_requests"], 15)
        self.assertEqual(st["queued_requests"], 0)
        self.assertIn(
            "IDLE_UTC=2000-01-01",
            out,
            "in-flight work must not reset the outstanding-work window",
        )

    def test_no_work_outstanding_RESETS_idle_utc(self):
        """The healthy gap between rungs: tokens paused, nothing in flight. The window resets every poll,
        so no amount of token staleness can ever accumulate into a halt."""
        st, out = self._emit(DYNAMO + IDLE_GAUGES, prev="198084", phase="generating")
        self.assertEqual(st["inflight_requests"], 0)
        self.assertNotIn("IDLE_UTC=2000-01-01", out)

    def test_unreachable_metrics_reset_idle_utc(self):
        """UNKNOWN in-flight must be treated as 'no evidence of work', never as work. Otherwise a dead
        /metrics endpoint would arm the halt on a perfectly healthy run."""
        st, out = self._emit(None, prev="198084", phase="generating")
        self.assertEqual(st["inflight_requests"], -1, "-1 == UNKNOWN, distinct from a measured 0")
        self.assertEqual(st["queued_requests"], -1)
        self.assertNotIn("IDLE_UTC=2000-01-01", out)

    def test_absent_inflight_gauge_is_unknown_and_resets_idle_utc(self):
        """A 200 with tokens but NO in-flight gauge (an engine that does not export one): still UNKNOWN."""
        st, out = self._emit(DYNAMO, prev="198084", phase="generating")
        self.assertEqual(st["inflight_requests"], -1)
        self.assertNotIn("IDLE_UTC=2000-01-01", out)

    # ---- status.json must always be valid JSON -----------------------------
    def test_reason_with_quotes_and_backslashes_stays_valid_json(self):
        r"""An unescaped `"` or `\` in any field emits invalid JSON. The governor then reads
        `jq -r '.state // ""'` as empty and drops the run from ALL supervision — silently. json.loads()
        succeeding IS the assertion here."""
        nasty = 'he said "boom" \\ and \\"more\\"'
        raw = self._emit_raw(DYNAMO + DYNAMO_GAUGES, prev="0", phase="generating", reason=nasty)
        try:
            st = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.fail(
                f"status.json is INVALID JSON ({exc}) → the governor's `jq -r '.state'` reads empty "
                f"and this run silently leaves ALL supervision. Emitted: {raw}"
            )
        self.assertEqual(st["reason"], nasty)
        self.assertEqual(st["state"], "running")

    def test_all_numeric_fields_are_json_numbers(self):
        st, _ = self._emit(DYNAMO + DYNAMO_GAUGES, prev="0", phase="generating")
        for k in (
            "progress_counter",
            "inflight_requests",
            "queued_requests",
            "expected_runtime_seconds",
        ):
            self.assertIsInstance(st[k], int, f"{k} must be a JSON number, not a string")

    def test_corrupt_progress_counter_file_cannot_break_json(self):
        """A torn/garbage .progress_counter must not splice a non-number into status.json."""
        st, _ = self._emit(DYNAMO + DYNAMO_GAUGES, prev='oops"', phase="generating")
        self.assertEqual(st["progress_counter"], 198084)


if __name__ == "__main__":
    unittest.main(verbosity=2)
