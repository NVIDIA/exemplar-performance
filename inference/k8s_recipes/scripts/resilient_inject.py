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

"""resilient_inject.py — SUBMIT-TIME manifest transformer for the disconnect-resilient `submit` path.

Reads an ALREADY-RENDERED, already-envsubst'd manifest stream on stdin and writes the transformed stream on
stdout. It is applied ONLY by scripts/submit.sh (the new detached path); the committed rendered/*.yaml are
never touched, so this injection does NOT roll any cell's recipe_hash (see the disconnect-resilient run contract
§2.1a floor + §4 Phase 1). The existing run.sh / deploy.sh / sweep.sh path is unaffected.

Two kinds:

  --kind bench   (the aiperf bench Job)
    * spec.activeDeadlineSeconds = <deadline>   (hard 2×-expected ceiling → Job Failed(DeadlineExceeded))
    * spec.ttlSecondsAfterFinished = <ttl>      (short → GC fires promptly → ownerRef cascade tears the
                                                  server down; §2.1a "short TTL bounds squat time")
    * labels llmb.nvidia.com/lifecycle=detached  on Job + pod template
    * env LLMB_CELL / LLMB_PROFILE / LLMB_SUBMIT_UTC / EXPECTED_RUNTIME_SECONDS / LLMB_HEALTH_MAX_ATTEMPTS
    * a volume + volumeMount for the shared per-namespace RWX **control** PVC (`llmb-control`) at /control
    * the container command is WRAPPED so the bench Job itself writes disconnect-durable state to the
      shared RWX CONTROL PVC — /control/<run-id>/_submit.json, /control/<run-id>/status.json
      (running→complete|failed, terminal reason completed|failed|timed-out), plus a backgrounded heartbeat
      loop that atomically (.tmp→mv) rewrites status.json every ~30s with heartbeat_utc / progress_counter /
      progress_utc / phase / inflight_requests / queued_requests / idle_utc / progress_note (the Phase-2
      governor's stall + timeout substrate — see the WRAPPER_TEMPLATE comment), and tees stdout+stderr to
      /control/<run-id>/logs/runner.log.
      The control PVC lives OUTSIDE the bench's ARTIFACTS_ROOT (=/artifacts/<run-id>), which the bench
      `rm -rf`s via CLEAN_ARTIFACTS_ROOT — so wrapper state is NEVER wiped mid-run (PHASE1-REVIEW F1) and,
      being RWX, is readable by status/logs concurrently while the bench holds the RWO artifacts PVC (F2).
      Bulk benchmark results still land on the per-cell RWO artifacts PVC (fetched post-run by collect).

  --kind server   (the vLLM server Deployment + Service)
    * metadata.ownerReferences = [Job <owner-name>/<owner-uid>]  on every Deployment/Service doc, so native
      GC cascade-deletes the server when the bench Job is GC'd (zero extra RBAC, no reaper — §2.1a #1).
      The owner UID is submit-time-only, which is exactly why this cannot live in the static manifest.

Usage (invoked by submit.sh; the wrapping/ordering is documented there):
  envsubst ... < bench-job.yaml | resilient_inject.py --kind bench \
      --run-id ID --cell CELL --profile PROF --namespace NS \
      --deadline-seconds N --ttl-seconds N --submit-utc STAMP
  envsubst ... < server.yaml    | resilient_inject.py --kind server \
      --owner-name <cell>-bench-<run-id> --owner-uid <job-uid>
"""

from __future__ import annotations

import argparse
import re
import sys

try:
    import yaml
except ImportError:
    sys.exit("resilient_inject: requires pyyaml")

LIFECYCLE_LABEL = "llmb.nvidia.com/lifecycle"
CONTROL_MOUNT = "/control"
CONTROL_VOLUME = "llmb-control"

# The in-pod wrapper runs under POSIX sh and writes logs and status to the shared control PVC.
# It publishes token progress and outstanding-request counts so the governor can distinguish a stalled
# engine from a normal idle period. Missing metrics remain unknown and never trigger cleanup.
WRAPPER_TEMPLATE = r"""# ==== llmb resilient wrapper (submit-time injected; NOT in the committed manifest) ====
set +e
_STATE_DIR="/control/${RUN_ID}"
mkdir -p "$_STATE_DIR/logs"
_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# submit record — makes run-id -> cell/namespace resolvable cold, after a fresh tsh login.
cat > "$_STATE_DIR/_submit.json" <<JSON
{"run_id":"${RUN_ID:-}","cell":"${LLMB_CELL:-}","profile":"${LLMB_PROFILE:-}","namespace":"${NAMESPACE:-}","recipe":"${RECIPE_SHORTNAME:-}","submitted_utc":"${LLMB_SUBMIT_UTC:-}"}
JSON
# main-owned signal file "<state> <phase> <reason>"; the heartbeat loop reflects it into status.json.
_SIGNAL="$_STATE_DIR/.state"
echo "running waiting-server " > "$_SIGNAL"
echo 0 > "$_STATE_DIR/.progress_counter"
_now > "$_STATE_DIR/.progress_utc"
_now > "$_STATE_DIR/.idle_utc"
_set_state() { echo "$1 $2 $3" > "$_SIGNAL"; }   # <state> <phase> <reason>
# Escape an arbitrary string for a JSON double-quoted scalar. status.json is assembled by a heredoc, and a
# raw `"` or `\` in ANY interpolated field (a reason, a cell path) produced INVALID JSON — whereupon the
# governor's `jq -r '.state // ""'` yields empty and that run is silently dropped from ALL supervision.
# Order matters: backslash first, then quote. Control characters are stripped; embedded newlines collapse.
_json_esc() { printf '%s' "${1:-}" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/[[:cntrl:]]//g' | tr -d '\n'; }
# Echo $1 when it is a non-negative integer, else the fallback $2. Guarantees every numeric field we splice
# into status.json is a valid JSON number, whatever a metrics endpoint or a stale state file contained.
_num_or() { case "${1:-}" in ''|*[!0-9]*) printf '%s\n' "$2" ;; *) printf '%s\n' "$1" ;; esac; }
# Fetch /metrics ONCE per heartbeat (progress + in-flight must describe the same instant). Prints the body,
# or prints NOTHING and returns 1. A curl that emits bytes and THEN fails (truncation) is a failure: a
# partial body would under-count and read as a stall. "I could not measure" must never look like "zero".
_metrics_fetch() {
  _m="$(curl -fsS --connect-timeout 3 --max-time 8 "${SERVER_URL:-}/metrics" 2>/dev/null)" || return 1
  [ -n "$_m" ] || return 1
  printf '%s\n' "$_m"
}
# Lane-agnostic progress signal, from a /metrics body on stdin. Sums each output/generation-token counter
# NAME across its label sets, then takes the MAX **across** names — never a cross-name sum. Two reasons:
#   * a Dynamo frontend re-exports the same tokens under both an aggregate and per-worker component names;
#     summing them double-counts, and a single worker restarting makes that sum DECREASE — which looks
#     exactly like a stall and would false-kill a healthy run.
#   * we do not need the true total, only a signal that ADVANCES while generation happens; the largest
#     single family provides that and is immune to any one family resetting.
# Prints nothing and returns 1 when no such counter exists (HTTP 200 with the series absent — a restarted
# frontend — is UNKNOWN, not zero). The historic vLLM-only pattern silently summed to 0 on Dynamo/SGLang.
_parse_tokens() {
  awk '
    /^[a-z_:]/ { nm = $0; sub(/[ {].*$/, "", nm)
                 if (nm ~ /(generation|output)_tokens(_total)?$/) { sum[nm] += $NF + 0; seen[nm] = 1 } }
    END { h = 0; m = 0
          for (k in seen) { if (h == 0 || sum[k] > m) { m = sum[k]; h = 1 } }
          if (h == 0) exit 1
          printf "%.0f\n", m }' 2>/dev/null
}
# Read a gauge from a /metrics body on stdin, by EXACT metric name (an allow-list, $1 space-separated).
# Exact names matter: a prefix match on `vllm:num_requests_waiting` would also swallow the per-reason
# breakdown `vllm:num_requests_waiting_by_reason` and double-count the queue. Same max-across-names rule as
# the token counter — Dynamo exports the same in-flight count at the frontend and the request plane.
# Returns 1 (prints nothing) when NONE of the names are present == UNKNOWN.
_parse_gauge() {
  awk -v _names="$1" '
    BEGIN { _n = split(_names, _a, " "); for (_i = 1; _i <= _n; _i++) want[_a[_i]] = 1 }
    /^[a-z_:]/ { nm = $0; sub(/[ {].*$/, "", nm)
                 if (nm in want) { sum[nm] += $NF + 0; seen[nm] = 1 } }
    END { h = 0; m = 0
          for (k in seen) { if (h == 0 || sum[k] > m) { m = sum[k]; h = 1 } }
          if (h == 0) exit 1
          printf "%.0f\n", m }' 2>/dev/null
}
# Requests the server currently has IN FLIGHT, and requests QUEUED behind them.
# The vLLM and Dynamo names are VERIFIED against committed results/*/server_metrics_surface.prom captures.
# The two `sglang:` names are UNVERIFIED — this repo holds no captured SGLang surface to check them against.
# They are safe to carry anyway because a wrong or absent name contributes nothing, which leaves the reading
# UNKNOWN, which DISARMS the halt: a bad guess here can only ever cost detection, never cause a false kill.
# Replace them the first time an SGLang cell writes a surface (do not guess further names in the meantime).
_parse_inflight() { _parse_gauge "vllm:num_requests_running dynamo_frontend_inflight_requests dynamo_request_plane_inflight_requests dynamo_frontend_active_requests sglang:num_running_reqs"; }
_parse_queued()   { _parse_gauge "vllm:num_requests_waiting dynamo_frontend_queued_requests sglang:num_queue_reqs"; }
# SINGLE writer of status.json — atomic .tmp→mv so cross-node (governor / status / logs) readers never tear.
_emit_status() {
  _st="$(cat "$_SIGNAL" 2>/dev/null)"; [ -n "$_st" ] || _st="running waiting-server "
  _state="$(printf '%s' "$_st" | cut -d' ' -f1)"
  _phase="$(printf '%s' "$_st" | cut -d' ' -f2)"
  _reason="$(printf '%s' "$_st" | cut -d' ' -f3-)"
  # progress_counter is a MONOTONIC HIGH-WATER MARK: it is the retained maximum, never the latest reading,
  # so a counter reset (frontend/worker restart) can never publish a regression that reads as a stall.
  _hw="$(_num_or "$(cat "$_STATE_DIR/.progress_counter" 2>/dev/null)" 0)"
  _inflight=-1; _queued=-1; _note=ok        # -1 == UNKNOWN. Never conflated with a measured 0.
  if _body="$(_metrics_fetch)"; then
    if _cur="$(printf '%s\n' "$_body" | _parse_tokens)" && [ -n "$_cur" ]; then
      if [ "$_cur" -gt "$_hw" ] 2>/dev/null; then
        _hw="$_cur"
        echo "$_hw" > "$_STATE_DIR/.progress_counter"; _now > "$_STATE_DIR/.progress_utc"
        # first real tokens seen while still waiting → we are now generating (gates the governor stall test).
        if [ "$_state" = running ] && [ "$_phase" = waiting-server ]; then
          _phase=generating; _set_state running generating "$_reason"
        fi
      fi
    else
      # HTTP 200 but the token series is ABSENT (frontend restarted / not exporting yet). UNKNOWN, not zero:
      # hold progress_utc fresh so an unmeasurable window can never be read as a stall.
      _note=metrics-no-token-counter; _now > "$_STATE_DIR/.progress_utc"
    fi
    _v="$(printf '%s\n' "$_body" | _parse_inflight)" && _inflight="$(_num_or "$_v" -1)"
    _v="$(printf '%s\n' "$_body" | _parse_queued)"   && _queued="$(_num_or "$_v" -1)"
  else
    _note=metrics-unreachable; _now > "$_STATE_DIR/.progress_utc"
  fi
  # The outstanding-work window. Reset it whenever work is ABSENT **or UNMEASURABLE**, so only a continuous
  # run of positively-observed in-flight requests can ever arm the governor's halt. This is the half that
  # makes arming the stall gate safe: a healthy gap between rungs (nothing in flight) and an unreadable
  # /metrics both keep resetting it, and neither can accumulate toward a kill.
  if [ "$_inflight" -gt 0 ] 2>/dev/null; then :; else _now > "$_STATE_DIR/.idle_utc"; fi
  # Surface an unmeasurable window in `reason` too when nothing more specific is set — visible, not silent.
  [ "$_note" = ok ] || [ -n "$_reason" ] || _reason="$_note"
  _pu="$(cat "$_STATE_DIR/.progress_utc" 2>/dev/null || _now)"
  _iu="$(cat "$_STATE_DIR/.idle_utc" 2>/dev/null || _now)"
  cat > "$_STATE_DIR/status.json.tmp" <<JSON
{"run_id":"$(_json_esc "${RUN_ID:-}")","cell":"$(_json_esc "${LLMB_CELL:-}")","profile":"$(_json_esc "${LLMB_PROFILE:-}")","namespace":"$(_json_esc "${NAMESPACE:-}")","recipe":"$(_json_esc "${RECIPE_SHORTNAME:-}")","state":"$(_json_esc "$_state")","phase":"$(_json_esc "$_phase")","reason":"$(_json_esc "$_reason")","heartbeat_utc":"$(_now)","progress_counter":$_hw,"progress_utc":"$_pu","progress_note":"$(_json_esc "$_note")","inflight_requests":$_inflight,"queued_requests":$_queued,"idle_utc":"$_iu","expected_runtime_seconds":$(_num_or "${EXPECTED_RUNTIME_SECONDS:-0}" 0),"updated_utc":"$(_now)"}
JSON
  mv "$_STATE_DIR/status.json.tmp" "$_STATE_DIR/status.json"
}
_emit_status                                   # publish status.json immediately (before generation)
_heartbeat() { while :; do sleep 30; _emit_status; done; }
_heartbeat & _hbpid=$!
# activeDeadlineSeconds / eviction sends SIGTERM to PID 1 (this shell). The bench runs BACKGROUNDED so the
# trap fires promptly (not deferred behind a foreground pipeline) → we reap the heartbeat, write a DURABLE
# terminal status (timed-out on SIGTERM, failed on SIGINT), best-effort kill the child, and exit.
_terminate() {  # <reason>
  kill "$_hbpid" 2>/dev/null; wait "$_hbpid" 2>/dev/null
  _set_state failed terminal "$1"; _emit_status
  [ -n "${_bpid:-}" ] && kill -TERM "$_bpid" 2>/dev/null
  echo "[llmb] runner terminated: reason=$1  status=$_STATE_DIR/status.json"
  exit 143
}
trap '_terminate timed-out' TERM
trap '_terminate failed' INT
{
  (
{SCRIPT}
  ); echo "$?" > "$_STATE_DIR/.exit_code"
} 2>&1 | tee -a "$_STATE_DIR/logs/runner.log" &
_bpid=$!
wait "$_bpid"
kill "$_hbpid" 2>/dev/null; wait "$_hbpid" 2>/dev/null
_rc="$(cat "$_STATE_DIR/.exit_code" 2>/dev/null || echo 1)"
if [ "$_rc" -eq 0 ] 2>/dev/null; then _set_state complete terminal completed; else _set_state failed terminal failed; fi
_emit_status
echo "[llmb] runner terminal: rc=$_rc  status=$_STATE_DIR/status.json  log=$_STATE_DIR/logs/runner.log"
exit "$_rc"
"""


def _health_attempts(health_timeout_seconds) -> int:
    # the bench health-wait sleeps 5s between probes → attempts = ceil(timeout / 5). Floor at 1.
    secs = int(health_timeout_seconds or 0)
    return max(1, (secs + 4) // 5)


def _inject_health_timeout(script: str) -> str:
    # F9: the bench's server-health wait is a hardcoded `MAX_ATTEMPTS=<n>  # <mins> min` (default 120×5s=600s),
    # far short of a COLD 550B checkpoint load (10–32 min) → the Job fails spuriously before the server is up.
    # Rewrite that assignment to honor the submit-time LLMB_HEALTH_MAX_ATTEMPTS env (default = the original n),
    # so --health-timeout can raise it without ever touching the committed rendered manifest (hash-neutral).
    return re.subn(
        r"(?m)^(\s*)MAX_ATTEMPTS=([0-9]+)([^\n]*)$",
        r"\1MAX_ATTEMPTS=${LLMB_HEALTH_MAX_ATTEMPTS:-\2}\3",
        script,
        count=1,
    )[0]


def _wrap_script(original: str) -> str:
    # Indent the original by 4 so it sits inside the `(` subshell; leading-whitespace-insensitive heredocs
    # (`<<'PY'` bodies) are unaffected because their terminators were already de-indented to column 0 by the
    # YAML block-scalar load, and dash matches `<<` terminators ignoring the surrounding indentation only when
    # they are at column 0 — so we keep the ORIGINAL text un-indented and instead open the subshell inline.
    return WRAPPER_TEMPLATE.replace("{SCRIPT}", _inject_health_timeout(original.rstrip("\n")))


def _ensure_labels(meta: dict) -> dict:
    labels = meta.setdefault("labels", {})
    labels[LIFECYCLE_LABEL] = "detached"
    return meta


def transform_bench(doc: dict, args) -> dict:
    spec = doc.setdefault("spec", {})
    spec["activeDeadlineSeconds"] = int(args.deadline_seconds)
    spec["ttlSecondsAfterFinished"] = int(args.ttl_seconds)
    _ensure_labels(doc.setdefault("metadata", {}))
    tmpl = spec.setdefault("template", {})
    _ensure_labels(tmpl.setdefault("metadata", {}))
    pod = tmpl.setdefault("spec", {})
    containers = pod.get("containers") or []
    if not containers:
        sys.exit("resilient_inject: bench Job has no containers to wrap")
    c = containers[0]
    # inject the wrapper's env inputs (RUN_ID/NAMESPACE/RECIPE_SHORTNAME/SERVER_URL already exist in the
    # template). EXPECTED_RUNTIME_SECONDS feeds the governor's 2×-median timeout; LLMB_HEALTH_MAX_ATTEMPTS is
    # the F9 cold-load server-health budget (attempts = health_timeout / 5s, computed by submit.sh).
    env = c.setdefault("env", [])
    have = {e.get("name") for e in env if isinstance(e, dict)}
    for name, val in (
        ("LLMB_CELL", args.cell),
        ("LLMB_PROFILE", args.profile),
        ("LLMB_SUBMIT_UTC", args.submit_utc),
        ("EXPECTED_RUNTIME_SECONDS", str(int(args.expected_runtime_seconds or 0))),
        ("LLMB_HEALTH_MAX_ATTEMPTS", str(_health_attempts(args.health_timeout))),
    ):
        if name not in have:
            env.append({"name": name, "value": val})
    # mount the shared per-namespace RWX control PVC at /control (wrapper state lives OUTSIDE the bench's
    # /artifacts ARTIFACTS_ROOT, which CLEAN_ARTIFACTS_ROOT wipes — F1; RWX → concurrent status/logs reads — F2)
    mounts = c.setdefault("volumeMounts", [])
    if not any(isinstance(m, dict) and m.get("name") == CONTROL_VOLUME for m in mounts):
        mounts.append({"name": CONTROL_VOLUME, "mountPath": CONTROL_MOUNT})
    vols = pod.setdefault("volumes", [])
    if not any(isinstance(v, dict) and v.get("name") == CONTROL_VOLUME for v in vols):
        vols.append(
            {
                "name": CONTROL_VOLUME,
                "persistentVolumeClaim": {"claimName": args.control_pvc},
            }
        )
    # wrap the command
    a = c.get("args")
    if not (isinstance(a, list) and a and isinstance(a[0], str)):
        sys.exit("resilient_inject: bench container args[0] is not the expected script string")
    a[0] = _wrap_script(a[0])
    return doc


def transform_server(doc: dict, args) -> dict:
    kind = doc.get("kind")
    # ConfigMap IS owned. The liveness-probe ConfigMap is per-RUN (name carries the leg tag) and is
    # useless the moment its server is gone, so without an ownerReference every run leaks one — seven
    # were found stranded in one namespace after a single night of sweeps, which is exactly the
    # "create without destroy" class check_create_destroy.py exists to catch.
    #
    # NOT every ConfigMap: a cell can also carry namespace-scoped singletons that are deliberately
    # re-applied and must SURVIVE the cascade (netscore-<run-id> is written precisely so a scored
    # headline outlives the run-owner GC). Ownership is therefore scoped to ConfigMaps the cell
    # renders as part of its server stack, which is what this transform is handed.
    if kind not in ("Deployment", "Service", "ConfigMap"):
        return doc  # PVC / RBAC etc. pass through untouched
    meta = doc.setdefault("metadata", {})
    meta["ownerReferences"] = [
        {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": args.owner_name,
            "uid": args.owner_uid,
            # controller=false: the Job is not the managing controller (no adoption). blockOwnerDeletion=false:
            # deleting the Job needs no extra RBAC on it. Both keep the cascade GC at the floor (§2.1a).
            "controller": False,
            "blockOwnerDeletion": False,
        }
    ]
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description="submit-time resilience injection")
    ap.add_argument("--kind", required=True, choices=["bench", "server"])
    # bench
    ap.add_argument("--run-id")
    ap.add_argument("--cell", default="")
    ap.add_argument("--profile", default="")
    ap.add_argument("--namespace", default="")
    ap.add_argument("--submit-utc", default="")
    ap.add_argument("--deadline-seconds", type=int)
    ap.add_argument("--ttl-seconds", type=int)
    ap.add_argument(
        "--control-pvc",
        default=CONTROL_VOLUME,
        help="name of the shared per-namespace RWX control PVC (default: llmb-control)",
    )
    ap.add_argument(
        "--expected-runtime-seconds",
        type=int,
        default=0,
        help="median full-sweep wall_seconds from runs.jsonl → governor 2×-median timeout",
    )
    ap.add_argument(
        "--health-timeout",
        type=int,
        default=2400,
        help="server-health wait budget (s) for a COLD big-model load (F9); attempts = /5",
    )
    # server
    ap.add_argument("--owner-name")
    ap.add_argument("--owner-uid")
    args = ap.parse_args()

    docs = list(yaml.safe_load_all(sys.stdin.read()))
    out = []
    for doc in docs:
        if not isinstance(doc, dict):
            out.append(doc)
            continue
        if args.kind == "bench" and doc.get("kind") == "Job":
            if args.deadline_seconds is None or args.ttl_seconds is None:
                sys.exit("resilient_inject --kind bench: --deadline-seconds and --ttl-seconds are required")
            out.append(transform_bench(doc, args))
        elif args.kind == "server":
            if not (args.owner_name and args.owner_uid):
                sys.exit("resilient_inject --kind server: --owner-name and --owner-uid are required")
            out.append(transform_server(doc, args))
        else:
            out.append(doc)

    yaml.safe_dump_all(out, sys.stdout, default_flow_style=False, sort_keys=False, width=100000)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
