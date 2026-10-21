# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

"""llmb_bench.smoke — validate a chat-completion smoke response. Pure.

Carved from bench-job.yaml.j2's inline `smoke_test_model` python: before a sweep, the runner sends one tiny
chat completion and hard-fails (exit 67) if the server isn't actually serving. This is that gate as a tested
function — the exact checks, in the exact order: parseable HTTP status → valid JSON → 2xx → no API error →
non-empty choices.
"""

from __future__ import annotations

import json

SMOKE_FAIL_EXIT = 67  # matches the shell's `raise SystemExit(67)` so the Job's failure code is unchanged


def validate_smoke(http_status, body_text: str) -> tuple[bool, str]:
    """(ok, reason) for a smoke chat-completion response. Mirrors bench-job.yaml.j2 exactly:
    non-numeric status / bad JSON / non-2xx / API error / missing choices → (False, why); else (True, note)."""
    try:
        code = int(http_status)
    except (ValueError, TypeError):
        return False, f"smoke request did not return an HTTP status: {http_status!r}"
    try:
        data = json.loads(body_text)
    except Exception as exc:
        return False, f"smoke response is not valid JSON: {exc}"
    if code < 200 or code >= 300:
        return False, f"smoke request failed HTTP {code}: {data}"
    if isinstance(data, dict) and data.get("error"):
        return False, f"smoke request returned API error: {data['error']}"
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        return False, f"smoke response missing choices: {data}"
    return True, f"smoke request passed HTTP {code}"
