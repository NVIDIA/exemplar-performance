#!/usr/bin/env bash
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

# shellcheck disable=SC1003,SC1090,SC2012,SC2015,SC2016,SC2034,SC2116,SC2207,SC2221,SC2222,SC2295,SC2317

# submit.sh <cell-dir> <cluster-profile> [run-id] [--rungs "<c1 c2 ...>"] [--dry-run]
#           [--deadline-seconds N] [--ttl-seconds N] [--health-timeout N]
#           [--repeat N | --variance N] [--stagger S] [--serial]
#
# Submit a detached run that persists control state and artifacts on Kubernetes PVCs.
# The bench Job owns the serving resources, so Kubernetes garbage collection releases them when the run ends.
# --repeat creates independent run IDs and consolidates their results without modifying committed recipes.
# Runtime injection leaves rendered manifests unchanged, preserving the recipe hash.
#
# Examples:
#   scripts/submit.sh <cell> <profile> --rungs "128"
#   scripts/submit.sh <cell> <profile> --dry-run
#   scripts/submit.sh <cell> <profile> --repeat 3
set -euo pipefail

RUNGS=""
DRY_RUN=0
DEADLINE_OVERRIDE=""
TTL_SECONDS="300"
HEALTH_OVERRIDE=""
ARGS=()
REPEAT=""
STAGGER="120"
SERIAL=0 # --repeat/--variance N variance-sweep orchestrator
while [ $# -gt 0 ]; do
    case "$1" in
        --rungs)
            RUNGS="${2:?--rungs needs a value}"
            shift 2
            ;;
        --rungs=*)
            RUNGS="${1#*=}"
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --deadline-seconds)
            DEADLINE_OVERRIDE="${2:?--deadline-seconds needs a value}"
            shift 2
            ;;
        --deadline-seconds=*)
            DEADLINE_OVERRIDE="${1#*=}"
            shift
            ;;
        --ttl-seconds)
            TTL_SECONDS="${2:?--ttl-seconds needs a value}"
            shift 2
            ;;
        --ttl-seconds=*)
            TTL_SECONDS="${1#*=}"
            shift
            ;;
        --health-timeout)
            HEALTH_OVERRIDE="${2:?--health-timeout needs a value}"
            shift 2
            ;;
        --health-timeout=*)
            HEALTH_OVERRIDE="${1#*=}"
            shift
            ;;
        --repeat | --variance)
            REPEAT="${2:?--repeat needs N}"
            shift 2
            ;;
        --repeat=* | --variance=*)
            REPEAT="${1#*=}"
            shift
            ;;
        --stagger)
            STAGGER="${2:?--stagger needs seconds}"
            shift 2
            ;;
        --stagger=*)
            STAGGER="${1#*=}"
            shift
            ;;
        --serial)
            SERIAL=1
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

CELL="${1:?usage: submit.sh <cell-dir> <cluster-profile> [run-id] [--rungs \"...\"] [--dry-run] [--deadline-seconds N] [--ttl-seconds N] [--health-timeout N]}"
PROFILE="${2:?need a cluster-profile name (cluster-profiles/<name>.env)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
# Resolve the driver type used by this recipe. Use the benchmark driver if the recipe has no lane metadata.
KIND="$(python3 "$ROOT/scripts/lane.py" "$CELL" kind 2> /dev/null || echo bench)"
BENCH="$CELL/rendered/${KIND}-job.yaml"
[ -f "$ENVF" ] || {
    echo "submit.sh: no profile at $ENVF" >&2
    exit 1
}
[ -f "$BENCH" ] || {
    echo "submit.sh: no $BENCH — run scripts/render.sh $CELL first" >&2
    exit 1
}
command -v envsubst > /dev/null || {
    echo "submit.sh: envsubst not found (install gettext)" >&2
    exit 1
}

set -a
. "$ENVF"
set +a # export the profile's cluster vars
# Resolve the model-cache claim only when the driver manifest mounts it. This keeps submission
# aligned with the install path while allowing workloads that do not use a model cache.
if grep -q 'MODEL_CACHE_PVC' "$BENCH"; then
    . "$ROOT/scripts/_model_cache.sh"
    llmb::resolve_model_cache_pvc "$CELL" "$ENVF" || exit 1
fi
kc() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }
# Normalize explicit and generated run IDs for Kubernetes labels and resource names.
if [ -n "${3:-}" ]; then
    export RUN_ID="$(python3 "$ROOT/scripts/run_id.py" "$CELL" --fit "$KIND" --label "$3" 2> /dev/null || echo "$3")"
else
    export RUN_ID="$(python3 "$ROOT/scripts/run_id.py" "$CELL" --fit "$KIND" 2> /dev/null || date -u +%Y%m%d-%H%M%S)"
fi
: "${OWNER:=}" "${CACHE_BUST:=}" "${DCGM_EXPORTER_URL:=}"
: "${BENCH_NODE_SELECTOR:=}" "${BENCH_CPU_REQUEST:=16}"
export OWNER CACHE_BUST DCGM_EXPORTER_URL BENCH_NODE_SELECTOR BENCH_CPU_REQUEST
# Provide optional substitutions used by some driver templates.
: "${NO_INTERNET_DNS_IP:=}" "${NO_INTERNET_KUBE_API_IP:=}"
export NO_INTERNET_DNS_IP NO_INTERNET_KUBE_API_IP
export LIVE_RUNGS="$RUNGS" # --rungs override (bench uses the CONCURRENCIES sed instead)

NAME="$(sed -n 's/^  name: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
[ -n "$NAME" ] || {
    echo "submit.sh: could not read envelope.name from $CELL/recipe.yaml" >&2
    exit 1
}
SCENARIO="$(sed -n 's/^  scenario: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
# Verify recipe and rendered resource identities before applying anything to the cluster.
if ! _ID_ERR="$(python3 "$(dirname "$0")/cell_identity.py" "$CELL" "$(dirname "$0")/../recipes" 2>&1 > /dev/null)"; then
    echo "submit.sh: FATAL — cell identity is inconsistent; refusing to apply anything." >&2
    printf '%s\n' "$_ID_ERR" >&2
    echo "  Fix: rename the cell in EVERY rendered/*.yaml as well as recipe.yaml (metadata.name must be" >&2
    echo "       <cell-name> or <cell-name>-*). Check offline, with no cluster:" >&2
    echo "       python3 scripts/cell_identity.py $CELL recipes" >&2
    exit 1
fi
# Build the Job name from the cell, lane, and run ID.
JOB_NAME="${NAME}-${KIND}-${RUN_ID}"
SUBMIT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Capture the recipe hash immediately before applying the Job; dry-run computes it without writing a receipt.
LAUNCH_RECEIPT="$ROOT/results/$RUN_ID/launch_attestation.json"
LAUNCH_HASH="$(python3 "$ROOT/scripts/recipe_hash.py" "$CELL" 2> /dev/null | sed 's/^recipe_hash: //')"
LAUNCH_CAPTURED_UTC="$SUBMIT_UTC"

_capture_launch() {
    python3 "$ROOT/scripts/launch_attestation.py" "$CELL" "$RUN_ID" --out "$LAUNCH_RECEIPT" > /dev/null
    LAUNCH_HASH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["recipe_hash"])' "$LAUNCH_RECEIPT")"
    LAUNCH_CAPTURED_UTC="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["captured_at_utc"])' "$LAUNCH_RECEIPT")"
}

# ============================================================================================================
# --repeat / --variance N : VARIANCE SWEEP orchestrator (additive; the single-run path below is UNTOUCHED).
# Branches here and EXITS — a leg never re-enters this block (it is submitted WITHOUT --repeat). See the header.
# ============================================================================================================
if [ -n "$REPEAT" ]; then
    # The variance sweep is a bench-only notion: it fans out name-suffixed scratch clones and consolidates the
    # legs back together by benchmark_id (`collect --sweep` → repro band). The "<name>-<tag>-bench-<tag>"
    # naming math below assumes the bench infix — so rather than silently produce a half-sweep, fail early.
    if [ "$KIND" != "bench" ]; then
        echo "submit.sh: --repeat/--variance is supported only for the llm-perf (bench) lane; this cell's lane is '$KIND'." >&2
        echo "submit.sh:   the variance sweep consolidates legs by benchmark_id (a bench-only repro notion)." >&2
        echo 'submit.sh:   submit a single detached run for this cell (drop --repeat), or use `run --detach`.' >&2
        exit 1
    fi
    case "$REPEAT" in '' | *[!0-9]*)
        echo "submit.sh: --repeat/--variance needs a positive integer (got '$REPEAT')" >&2
        exit 1
        ;;
    esac
    [ "$REPEAT" -ge 1 ] || {
        echo "submit.sh: --repeat/--variance must be >= 1" >&2
        exit 1
    }
    case "$STAGGER" in '' | *[!0-9]*)
        echo "submit.sh: --stagger needs an integer number of seconds (got '$STAGGER')" >&2
        exit 1
        ;;
    esac

    SWEEP_SALT="$(printf '%06x' "$((($$ ^ $(date +%s)) % 16777216))")"
    SWEEP_ID="sw$(date -u +%Y%m%dt%H%M%S)${SWEEP_SALT}"
    # Per-leg tag is used as BOTH the name-suffix AND the run-id, so the bench Job name
    # "<NAME>-<tag>-bench-<tag>" = len(NAME)+8+2·len(tag) must be ≤63 (DNS-1123). Size the tag to fit; the
    # shared salt gives cross-sweep uniqueness, the leg index gives per-leg distinctness. (Same intrinsic
    # name-length limit as scripts/parallel_repro.sh — a very long envelope.name simply can't fan out.)
    TAG_BUDGET=$(((63 - ${#NAME} - 8) / 2))
    if [ "$TAG_BUDGET" -lt 3 ]; then
        echo "submit.sh: envelope.name '$NAME' (${#NAME} chars) is too long for a --repeat variance sweep —" >&2
        echo "submit.sh:   only room for a ${TAG_BUDGET}-char per-leg tag before the bench/server names exceed 63." >&2
        echo "submit.sh:   shorten envelope.name for variance sweeps (same limit as scripts/parallel_repro.sh)." >&2
        exit 1
    fi
    SWEEP_MODE=$([ "$SERIAL" = 1 ] && echo serial || echo parallel)
    ORIG_BID="$(python3 "$ROOT/scripts/benchmark_id.py" "$CELL" --short 2> /dev/null || echo '?')"

    # Durable scratch root under the gitignored results/ (survives until `collect --sweep`); /tmp for --dry-run.
    if [ "$DRY_RUN" = 1 ]; then
        SWEEP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/llmb-sweep-${SWEEP_ID}-XXXX")"
    else
        SWEEP_ROOT="$ROOT/results/.sweeps/$SWEEP_ID"
        mkdir -p "$SWEEP_ROOT/legs"
    fi

    # single-run flags to FORWARD to each leg (never --repeat/--stagger/--serial).
    leg_flags=()
    [ -n "$RUNGS" ] && leg_flags+=(--rungs "$RUNGS")
    [ -n "$DEADLINE_OVERRIDE" ] && leg_flags+=(--deadline-seconds "$DEADLINE_OVERRIDE")
    [ "$TTL_SECONDS" != "300" ] && leg_flags+=(--ttl-seconds "$TTL_SECONDS")
    [ -n "$HEALTH_OVERRIDE" ] && leg_flags+=(--health-timeout "$HEALTH_OVERRIDE")

    # 1) clone N name-SUFFIXED scratch copies (distinct k8s objects → collision-free) + re-render (parallel_repro trick).
    leg_tags=()
    leg_names=()
    leg_cells=()
    for i in $(seq 1 "$REPEAT"); do
        idxstr="$(printf '%x' "$i")" # leg index (hex → DNS-safe, per-leg distinct).
        salt_len=$((TAG_BUDGET - 1 - ${#idxstr}))
        [ "$salt_len" -gt 6 ] && salt_len=6
        if [ "$salt_len" -lt 1 ]; then
            echo "submit.sh: --repeat $REPEAT needs more tag room than envelope.name '$NAME' allows (leg $i)." >&2
            rm -rf "${SWEEP_ROOT}"
            exit 1
        fi
        tag="r${idxstr}${SWEEP_SALT:0:salt_len}" # DNS-1123-safe, unique per leg; also the leg run-id.
        cell_name="${NAME}-${tag}"
        server_name="${cell_name}-server"
        bench_name="${cell_name}-bench-${tag}"
        if [ "${#server_name}" -gt 63 ] || [ "${#bench_name}" -gt 63 ]; then
            echo "submit.sh: --repeat generated a name >63-char k8s limit (server=${#server_name} bench=${#bench_name})." >&2
            echo "submit.sh:   shorten envelope.name '$NAME' for variance sweeps." >&2
            rm -rf "${SWEEP_ROOT}"
            exit 1
        fi
        dst="$SWEEP_ROOT/legs/$tag"
        mkdir -p "$(dirname "$dst")"
        cp -R "$CELL" "$dst"
        rm -rf "$dst/rendered" "$dst/runs.jsonl" "$dst/RESULTS.md" "$dst/record.json" "$dst/results" 2> /dev/null || true
        python3 - "$dst/recipe.yaml" "$tag" << 'PY'
import yaml, sys
p, tag = sys.argv[1], sys.argv[2]
d = yaml.safe_load(open(p))
d["envelope"]["name"] = f'{d["envelope"]["name"]}-{tag}'
yaml.safe_dump(d, open(p, "w"), sort_keys=False)
PY
        bash "$ROOT/scripts/render.sh" "$dst" > /dev/null
        leg_bid="$(python3 "$ROOT/scripts/benchmark_id.py" "$dst" --short 2> /dev/null || echo '?')"
        if [ "$leg_bid" != "$ORIG_BID" ]; then
            echo "submit.sh: FATAL — leg $tag benchmark_id ($leg_bid) != original ($ORIG_BID); would not consolidate." >&2
            rm -rf "${SWEEP_ROOT}"
            exit 1
        fi
        leg_tags+=("$tag")
        leg_names+=("$cell_name")
        leg_cells+=("$dst")
    done

    # cluster-side durable sweep record: no ownerRef/TTL → survives Job GC (cold cross-machine `collect --sweep`).
    render_sweep_cm() {
        local ids="${leg_tags[*]}"
        cat << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: llmb-sweep-${SWEEP_ID}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/managed-by: llmb-recipe
    llmb.nvidia.com/sweep-id: ${SWEEP_ID}
    llmb.nvidia.com/lifecycle: detached
data:
  sweep_id: "${SWEEP_ID}"
  cell: "${CELL}"
  profile: "${PROFILE}"
  namespace: "${NAMESPACE}"
  recipe: "${NAME}"
  repeat: "${REPEAT}"
  mode: "${SWEEP_MODE}"
  stagger_seconds: "${STAGGER}"
  benchmark_id: "${ORIG_BID}"
  run_ids: "${ids}"
  created_utc: "${SUBMIT_UTC}"
EOF
    }
    # local durable sweep record: run-ids + original cell + each leg's scratch cell dir (built via python for JSON safety).
    write_sweep_record() { # <out-path>
        python3 - "$1" "$SWEEP_ID" "$CELL" "$PROFILE" "${NAMESPACE:-}" "$NAME" "$REPEAT" "$SWEEP_MODE" \
            "$STAGGER" "$ORIG_BID" "$SUBMIT_UTC" "$SWEEP_ROOT" "${leg_tags[@]}" << 'PY'
import json, sys
out, sweep_id, cell, profile, ns, recipe, repeat, mode, stagger, bid, created, root = sys.argv[1:13]
tags = sys.argv[13:]
rec = {
    "sweep_id": sweep_id, "cell": cell, "profile": profile, "namespace": ns, "recipe": recipe,
    "repeat": int(repeat), "mode": mode, "stagger_seconds": int(stagger),
    "benchmark_id": bid, "created_utc": created, "run_ids": tags,
    # Track both the requested leg tag and the run ID returned by the submit path.
    "legs": [{"run_id": t, "tag": t, "submitted_run_id": None, "cell_name": f"{recipe}-{t}",
              "scratch_name": f"{recipe}-{t}", "scratch_cell": f"{root}/legs/{t}"} for t in tags],
}
open(out, "w").write(json.dumps(rec, indent=2) + "\n")
PY
    }

    # ---- DRY RUN: OFFLINE. Show the N distinct-named legs (IDENTICAL benchmark_id), the stagger plan, the sweep
    #      record; run each leg's OWN resilient submit --dry-run; apply/submit NOTHING; drop the scratch. ---------
    if [ "$DRY_RUN" = 1 ]; then
        echo "=== SUBMIT --repeat $REPEAT — VARIANCE SWEEP  (sweep $SWEEP_ID) → profile=$PROFILE  (NOTHING APPLIED) ==="
        echo "  sweep-id:            $SWEEP_ID"
        echo "  original cell:       $NAME   ($CELL)"
        echo "  benchmark_id (orig): $ORIG_BID   (EXCLUDES envelope.name → every leg matches → consolidates as repeats)"
        echo "  mode:                $SWEEP_MODE"
        echo "  stagger:             ${STAGGER}s between leg cold-starts (shared model-cache NFS weight-load contention guard)"
        echo "  legs (N=$REPEAT):"
        for idx in "${!leg_tags[@]}"; do
            if [ "$SWEEP_MODE" = serial ]; then
                start=$([ "$idx" = 0 ] && echo "+0s" || echo "after leg $idx completes")
            else
                start="+$((idx * STAGGER))s"
            fi
            printf '    #%d  name=%s  run-id=%s  bench=%s-bench-%s  benchmark_id=%s  start %s\n' \
                "$((idx + 1))" "${leg_names[$idx]}" "${leg_tags[$idx]}" "${leg_names[$idx]}" "${leg_tags[$idx]}" "$ORIG_BID" "$start"
        done
        echo ""
        echo "  collect: llmb-k8s collect --sweep $SWEEP_ID [--cluster $PROFILE]   (fetch N legs → consolidate → compare --repro band)"
        echo "  fleet:   llmb-k8s fleet --watch   (live all-clusters, active-runs-first status pane)"
        echo ""
        echo "# ================= sweep record ConfigMap (llmb-sweep-<id>; no ownerRef/TTL → survives Job GC) ========"
        echo "---"
        render_sweep_cm
        echo ""
        echo "# ================= per-leg resilient submit (full injection; identical to single-run submit) =========="
        echo "# NOTE: each leg is a full detached resilient submit — ownerRef teardown + RWX control PVC + index CM."
        for idx in "${!leg_tags[@]}"; do
            if bash "$ROOT/scripts/submit.sh" "${leg_cells[$idx]}" "$PROFILE" "${leg_tags[$idx]}" --dry-run \
                ${leg_flags[@]+"${leg_flags[@]}"} > /dev/null 2>&1; then
                echo "  [leg #$((idx + 1)) ${leg_tags[$idx]}] resilient submit --dry-run OK (bench Job + server ownerRef→Job + control PVC + index CM)"
            else
                echo "  [leg #$((idx + 1)) ${leg_tags[$idx]}] resilient submit --dry-run FAILED" >&2
            fi
        done
        echo ""
        echo "# (dry-run complete — scratch under ${SWEEP_ROOT} is EPHEMERAL; no PVC created, nothing applied, no cluster mutation)"
        rm -rf "$SWEEP_ROOT"
        exit 0
    fi

    # ---- LIVE SWEEP: record durably FIRST, then detach-submit each leg staggered (or serial). -------------------
    echo "submit: ▶ variance sweep $SWEEP_ID — ${REPEAT}× $NAME on $PROFILE (mode=$SWEEP_MODE, stagger=${STAGGER}s)"
    SWEEPS_DIR="$ROOT/results/.sweeps"
    mkdir -p "$SWEEPS_DIR"
    write_sweep_record "$SWEEPS_DIR/${SWEEP_ID}.json" # Write the sweep handle before submitting legs.
    render_sweep_cm | kc apply -f - \
        || echo "submit: WARN — sweep ConfigMap apply failed (local record at $SWEEPS_DIR/${SWEEP_ID}.json still written)" >&2

    legs_ok=0
    for idx in "${!leg_tags[@]}"; do
        tag="${leg_tags[$idx]}"
        dst="${leg_cells[$idx]}"
        echo "submit: ── leg #$((idx + 1))/$REPEAT  run-id=$tag  (${leg_names[$idx]})"
        if bash "$ROOT/scripts/submit.sh" "$dst" "$PROFILE" "$tag" ${leg_flags[@]+"${leg_flags[@]}"}; then
            legs_ok=$((legs_ok + 1))
            # Bind tag -> minted run-id in the sweep record NOW, while we can see the leg's own submit
            # record. Without this the mapping exists only as a path substring and collect --sweep has to
            # infer it. Best-effort: a failure here degrades to that inference, never to a failed submit.
            python3 - "$SWEEPS_DIR/${SWEEP_ID}.json" "$tag" "$dst" << 'PY' || true
import json, pathlib, sys
rec_p, tag, legdir = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3].rstrip("/")
subs = pathlib.Path(__file__).resolve().parent  # placeholder; real dir resolved below
subs = rec_p.parent.parent / ".submits"
rid = ""
if subs.is_dir():
    for f in sorted(subs.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try: d = json.loads(f.read_text())
        except Exception: continue
        if (d.get("cell") or "").rstrip("/") == legdir and d.get("run_id"):
            rid = d["run_id"]; break
if rid and rec_p.is_file():
    rec = json.loads(rec_p.read_text())
    for leg in rec.get("legs", []):
        if leg.get("tag") == tag or leg.get("run_id") == tag:
            leg["submitted_run_id"] = rid
    rec_p.write_text(json.dumps(rec, indent=2) + "\n")
PY
        else
            echo "submit: WARN — leg $tag submit failed; continuing (its run-id stays in the sweep record)." >&2
        fi
        if [ "$idx" -lt "$((REPEAT - 1))" ]; then # stagger / serialize between legs (not after the last).
            if [ "$SWEEP_MODE" = serial ]; then
                echo "submit: --serial — waiting for leg $tag Job to finish before launching the next…"
                # Resolve by labels because the generated run ID may differ from the requested leg tag.
                legjob=""
                for _ in $( # the leg submit is detached; the Job may lag.
                    seq 1 30
                ); do
                    legjob="$(kc -n "$NAMESPACE" get job -l "llmb.nvidia.com/cell=${leg_names[$idx]}" \
                        -o jsonpath='{.items[0].metadata.name}' 2> /dev/null)" || legjob=""
                    [ -n "$legjob" ] && break
                    sleep 2
                done
                if [ -z "$legjob" ]; then
                    # Stop if the submitted Job cannot be resolved.
                    echo "submit: ERROR — --serial cannot find leg $tag's bench Job (label llmb.nvidia.com/cell=${leg_names[$idx]})." >&2
                    echo "submit:   Refusing to silently continue in PARALLEL — that is what --serial exists to prevent." >&2
                    echo "submit:   Legs already submitted: ${leg_tags[*]:0:$((idx + 1))}  ·  collect with: llmb-k8s collect --sweep $SWEEP_ID" >&2
                    exit 1
                fi
                echo "submit:   leg $tag -> job/$legjob"
                # Poll for EITHER terminal state. `wait --for=condition=complete` alone would block the full
                # timeout on a leg that FAILED. A read is trusted only when it succeeds AND returns something.
                while :; do
                    jst="$(kc -n "$NAMESPACE" get job "$legjob" \
                        -o jsonpath='{.status.succeeded}|{.status.failed}' 2> /dev/null)" || {
                        sleep 15
                        continue
                    }
                    case "$jst" in
                        '' | '|')
                            sleep 15
                            continue
                            ;; # unreadable/blank -> not evidence of anything
                    esac
                    case "$jst" in
                        *[1-9]*)
                            echo "submit:   leg $tag terminal (succeeded|failed = $jst)"
                            break
                            ;;
                        *) sleep 15 ;;
                    esac
                done
            else
                echo "submit: staggering s before launching the next model load…"
                sleep "$STAGGER"
            fi
        fi
    done

    # Treat any partial submission as an error.
    if [ "$legs_ok" -lt "$REPEAT" ]; then
        cat >&2 << EOF

submit: ❌ variance sweep $SWEEP_ID — only $legs_ok/$REPEAT legs submitted.
$([ "$legs_ok" = 0 ] && echo "        NOTHING WAS SUBMITTED. No GPU work started; there is nothing to collect.")
        Re-run after fixing the cause; already-submitted legs (if any) remain collectable.

  sweep-id: $SWEEP_ID
  legs:     ${leg_tags[*]}
  collect:  llmb-k8s collect --sweep $SWEEP_ID
EOF
        exit 1
    fi
    cat << EOF

submit: ✅ variance sweep $SWEEP_ID launched — $legs_ok/$REPEAT legs submitted (each independently disconnect-durable).
        You may disconnect (laptop + tsh) now. Each leg tears its own server down via ownerReference GC.

  sweep-id: $SWEEP_ID
  legs:     ${leg_tags[*]}
  collect:  llmb-k8s collect --sweep $SWEEP_ID
EOF
    exit 0
fi

# Set the deadline from an explicit override, recipe expectation, or scenario default.
if [ -n "$DEADLINE_OVERRIDE" ]; then
    DEADLINE_SECONDS="$DEADLINE_OVERRIDE"
else
    DEADLINE_SECONDS="$(
        python3 - "$CELL/recipe.yaml" "$SCENARIO" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
scenario = sys.argv[2]
env = r.get("envelope") or {}
exp = ((r.get("bench") or {}).get("expected_runtime_min")
       or (r.get("replay") or {}).get("expected_runtime_min")
       or env.get("expected_runtime_min"))
print(int(float(exp) * 60 * 2) if exp else 36000)
PY
    )"
fi

# ---- server-health wait budget ------------------------------------------------------------------
# Use the recipe server startup budget for the benchmark health wait; --health-timeout overrides it.
if [ -n "$HEALTH_OVERRIDE" ]; then
    HEALTH_TIMEOUT="$HEALTH_OVERRIDE"
else
    HEALTH_TIMEOUT="$(
        python3 - "$CELL/recipe.yaml" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
s = (r.get("serving") or {}).get("startup_timeout_s")
try: v = int(float(s)) if s else 2400
except Exception: v = 2400
print(max(1800, v))
PY
    )"
fi

# Estimate runtime from completed local run records. Zero leaves enforcement to activeDeadlineSeconds.
EXPECTED_RUNTIME_SECONDS="$(
    python3 - "$CELL/runs.jsonl" << 'PY'
import json, statistics, sys
durs = []
try:
    for line in open(sys.argv[1]):
        try: r = json.loads(line)
        except Exception: continue
        w = r.get("wall_seconds")
        if isinstance(w, (int, float)) and w > 0 and ("metric" in r or "value" in r or r.get("gpu_count")):
            durs.append(float(w))
except FileNotFoundError:
    pass
print(int(statistics.median(durs)) if durs else 0)
PY
)"

# ---- shared per-namespace RWX control PVC -------------
CONTROL_PVC="llmb-control"
: "${CONTROL_SIZE:=5Gi}"

# Substitute only cluster and runtime variables so in-container shell variables remain intact.
WL='$NAMESPACE $RUN_ID $OWNER $IMAGE_PULL_SECRET $MODEL_CACHE_PVC $DCGM_EXPORTER_URL $CACHE_BUST $BENCH_NODE_SELECTOR $BENCH_CPU_REQUEST'

# Render the driver Job and inject lifecycle metadata and optional concurrency rungs.
render_bench() {
    if [ "$KIND" = "bench" ] && [ -n "$RUNGS" ]; then
        case "$RUNGS" in
            *[!0-9\ ]*)
                echo "submit.sh: --rungs must be space-separated integers (got: '$RUNGS')" >&2
                exit 1
                ;;
        esac
        envsubst "$WL" < "$BENCH" \
            | sed "s/\(name: CONCURRENCIES, value: \)\"[^\"]*\"/\1\"$RUNGS\"/"
    else
        envsubst "$WL" < "$BENCH"
    fi | python3 "$ROOT/scripts/resilient_inject.py" --kind bench \
        --run-id "$RUN_ID" --cell "$CELL" --profile "$PROFILE" --namespace "${NAMESPACE:-}" \
        --submit-utc "$SUBMIT_UTC" --deadline-seconds "$DEADLINE_SECONDS" --ttl-seconds "$TTL_SECONDS" \
        --control-pvc "$CONTROL_PVC" --expected-runtime-seconds "$EXPECTED_RUNTIME_SECONDS" \
        --health-timeout "$HEALTH_TIMEOUT"
}

# Render the shared RWX control PVC used for live status and logs; bulk results stay on the artifacts PVC.
render_control_pvc() {
    echo "apiVersion: v1"
    echo "kind: PersistentVolumeClaim"
    echo "metadata:"
    echo "  name: ${CONTROL_PVC}"
    echo "  namespace: ${NAMESPACE}"
    echo "  labels: { app.kubernetes.io/managed-by: llmb-recipe, app.kubernetes.io/component: control }"
    echo "spec:"
    echo "  accessModes: [ReadWriteMany]"
    [ -n "${CONTROL_STORAGE_CLASS:-}" ] && echo "  storageClassName: ${CONTROL_STORAGE_CLASS}"
    echo "  resources: { requests: { storage: ${CONTROL_SIZE} } }"
}

# Persist a namespace-scoped run index until successful collection removes it.
render_index_cm() {
    cat << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: llmb-submit-${RUN_ID}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/managed-by: llmb-recipe
    llmb.nvidia.com/run-id: ${RUN_ID}
    llmb.nvidia.com/lifecycle: detached
data:
  run_id: "${RUN_ID}"
  cell: "${CELL}"
  profile: "${PROFILE}"
  namespace: "${NAMESPACE}"
  recipe: "${NAME}"
  recipe_hash_at_launch: "${LAUNCH_HASH}"
  recipe_hash_captured_at_utc: "${LAUNCH_CAPTURED_UTC}"
  artifacts_pvc: "${NAME}-artifacts"
  job_name: "${JOB_NAME}"
  submitted_utc: "${SUBMIT_UTC}"
EOF
}

# Render the server stack (every rendered/*.yaml except the Job manifests) with an ownerReference→Job.
render_server() {
    local owner_uid="$1" f wl
    for f in "$CELL"/rendered/*.yaml; do
        [ -e "$f" ] || continue
        # skip every lane's driver Job manifest (only one exists per cell) — it is applied by render_bench, owned
        # by nothing; the server stack is everything ELSE (Deployment/Service/PVC/ConfigMap) owned by that Job.
        case "$(basename "$f")" in bench-job.yaml) continue ;; esac
        wl=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*\}' "$f" | tr -d '${}' | sort -u | sed 's/^/$/' | paste -sd' ' -)
        echo "---"
        envsubst "$wl" < "$f" \
            | python3 "$ROOT/scripts/resilient_inject.py" --kind server \
                --owner-name "$JOB_NAME" --owner-uid "$owner_uid"
    done
}

# ---- DRY RUN: OFFLINE. Show the injected ownerReference + deadline + TTL + labels; apply nothing. --------
if [ "$DRY_RUN" = 1 ]; then
    echo "=== SUBMIT --dry-run: $NAME → profile=$PROFILE (NOTHING APPLIED) ==="
    echo "  run-id:              $RUN_ID"
    echo "  ${KIND} Job:$(printf '%*s' $((13 - ${#KIND})) '')$JOB_NAME"
    echo "  namespace:           ${NAMESPACE:-<unset>}"
    echo "  scenario:            ${SCENARIO:-<unset>}"
    echo "  activeDeadlineSeconds: $DEADLINE_SECONDS   (2× expected; source: ${DEADLINE_OVERRIDE:+--deadline-seconds override}${DEADLINE_OVERRIDE:-per-scenario default / expected_runtime_min})"
    echo "  ttlSecondsAfterFinished: $TTL_SECONDS   (short → prompt GC cascade)"
    echo "  health-timeout:      ${HEALTH_TIMEOUT}s   (cold-load server-health budget; source: ${HEALTH_OVERRIDE:+--health-timeout override}${HEALTH_OVERRIDE:-serving.startup_timeout_s, floor 1800s})"
    echo "  expected_runtime:    ${EXPECTED_RUNTIME_SECONDS}s   (median full-sweep wall_seconds from runs.jsonl → governor 2×-median timeout; 0 = native ADS backstop)"
    echo "  control PVC:         ${CONTROL_PVC}  (ReadWriteMany, class=${CONTROL_STORAGE_CLASS:-<unset — REQUIRED for live submit>}, ${CONTROL_SIZE})  → wrapper state at /control/${RUN_ID}/{status.json,logs/runner.log,_submit.json}"
    echo "  artifacts PVC:       ${NAME}-artifacts  (ReadWriteOnce → bulk results only, fetched post-run by collect)"
    echo ""
    echo "# ================= shared RWX control PVC (llmb-control — wrapper state; bench NEVER wipes it) ======="
    echo "---"
    render_control_pvc
    echo ""
    echo "# ================= bench Job (deadline/TTL/labels + /control-mount log+status wrapper injected) ======"
    echo "# NOTE: the bench pipeline is BACKGROUNDED + wait'ed so the deadline TERM trap fires (reason=timed-out);"
    echo "#       wrapper state is under /control/<run-id>/ (RWX, bench-clean-safe); mounts PVC '${CONTROL_PVC}'."
    render_bench
    echo ""
    echo "# ================= server stack (ownerReference → bench Job injected) ================================"
    echo "# NOTE: --dry-run uses a PLACEHOLDER owner uid; at real submit the bench Job's live .metadata.uid is used."
    render_server "00000000-0000-0000-0000-000000000000-DRYRUN-PLACEHOLDER"
    echo ""
    echo "# ================= run-id → cell INDEX ConfigMap (no ownerRef, no TTL → survives Job GC) ============="
    echo "---"
    render_index_cm
    echo ""
    echo "# (dry-run complete — no PVC created, nothing applied)"
    exit 0
fi

# ---- LIVE SUBMIT ------------------------------------------------------------------------------------------
# Stage-only preflight validates cache access, integrity, secrets, architecture, and RDMA before submission.
_pf_out="$(mktemp)"
_pf_rc=0
set +e
python3 "$ROOT/scripts/preflight.py" "$CELL" "$PROFILE" --stage-only 2>&1 | tee "$_pf_out"
_pf_rc=${PIPESTATUS[0]}
set -e
if [ "$_pf_rc" -ne 0 ]; then
    echo "submit: compatibility preflight failed — nothing was applied:" >&2
    cat "$_pf_out" >&2
    rm -f "$_pf_out"
    exit 1
fi
rm -f "$_pf_out"

# Detached status readers require a ReadWriteMany control PVC.
if [ -z "${CONTROL_STORAGE_CLASS:-}" ]; then
    echo "submit: FATAL — CONTROL_STORAGE_CLASS is unset in profile '$PROFILE'." >&2
    echo "submit:   The disconnect-resilient path needs a shared RWX (ReadWriteMany) 'control' storage class" >&2
    echo "submit:   (for example, a provider-supported RWX class). Add CONTROL_STORAGE_CLASS to" >&2
    echo "submit:   cluster-profiles/${PROFILE}.env (see cluster-profiles/README.md), then re-submit." >&2
    exit 1
fi

# Ensure the shared per-namespace RWX control PVC (idempotent — one per namespace, shared across all runs).
if ! kc -n "$NAMESPACE" get pvc "$CONTROL_PVC" > /dev/null 2>&1; then
    echo "submit: creating shared RWX control PVC $CONTROL_PVC (class=$CONTROL_STORAGE_CLASS, $CONTROL_SIZE)"
    render_control_pvc | kc apply -f -
fi

# Ensure the per-cell artifacts PVC (idempotent; RWO — bulk results only).
if ! kc -n "$NAMESPACE" get pvc "${NAME}-artifacts" > /dev/null 2>&1; then
    echo "submit: creating artifacts PVC ${NAME}-artifacts"
    {
        echo "apiVersion: v1"
        echo "kind: PersistentVolumeClaim"
        echo "metadata: { name: ${NAME}-artifacts, namespace: ${NAMESPACE} }"
        echo "spec:"
        echo "  accessModes: [ReadWriteOnce]"
        [ -n "${ARTIFACTS_STORAGE_CLASS:-}" ] && echo "  storageClassName: ${ARTIFACTS_STORAGE_CLASS}"
        echo "  resources: { requests: { storage: ${ARTIFACTS_SIZE:-20Gi} } }"
    } | kc apply -f -
fi

# Collision guard: refuse to launch a second active bench Job for this cell (two loads corrupt both results).
_active=$(kc -n "$NAMESPACE" get jobs -l "llmb.nvidia.com/cell=${NAME},app.kubernetes.io/component!=run-owner" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\n"}{end}' 2> /dev/null \
    | awk -F'\t' '($2+0)>0 {printf "%s ", $1}' || true)
if [ -n "$_active" ]; then
    echo "submit: BLOCKED — an active bench job for '$NAME' is already running: $_active" >&2
    echo "submit:   Two concurrent sweeps against the same server corrupt both results." >&2
    exit 1
fi

SUBMITS_DIR="$ROOT/results/.submits"
mkdir -p "$SUBMITS_DIR"
_write_local_record() { # <job-uid> — durable local run handle
    cat > "$SUBMITS_DIR/${RUN_ID}.json" << JSON
{
  "run_id": "$RUN_ID",
  "cell": "$CELL",
  "profile": "$PROFILE",
  "namespace": "${NAMESPACE:-}",
  "recipe": "$NAME",
  "job_name": "$JOB_NAME",
  "job_uid": "$1",
  "artifacts_pvc": "${NAME}-artifacts",
  "control_pvc": "$CONTROL_PVC",
  "deadline_seconds": $DEADLINE_SECONDS,
  "ttl_seconds": $TTL_SECONDS,
  "health_timeout_seconds": $HEALTH_TIMEOUT,
  "expected_runtime_seconds": $EXPECTED_RUNTIME_SECONDS,
  "recipe_hash_at_launch": "$LAUNCH_HASH",
  "recipe_hash_captured_at_utc": "$LAUNCH_CAPTURED_UTC",
  "submitted_utc": "$SUBMIT_UTC"
}
JSON
}
# Delete a partially submitted Job if the server stack cannot be applied.
_abort_orphan() { # <message>
    echo "submit: FATAL — $1" >&2
    echo "submit:   deleting the orphaned bench Job job/$JOB_NAME so it does not run against a missing server." >&2
    kc -n "$NAMESPACE" delete job "$JOB_NAME" --ignore-not-found --wait=false > /dev/null 2>&1 || true
    echo "submit:   local run-id handle kept at $SUBMITS_DIR/${RUN_ID}.json (submit was aborted; nothing to collect)." >&2
    exit 1
}

# Apply the bench Job first so its UID can own the server resources.
echo "submit: apply ${KIND} Job  (cell=$NAME  run-id=$RUN_ID  ns=$NAMESPACE  deadline=${DEADLINE_SECONDS}s  ttl=${TTL_SECONDS}s  health-wait=${HEALTH_TIMEOUT}s)"
if [ -n "$RUNGS" ]; then
    echo "submit: ⚡ RUNGS OVERRIDE -> \"$RUNGS\" (functional smoke; committed sweep unchanged, NOT publishable)"
fi
_capture_launch
render_bench | kc apply -f -

# Read the Job UID with bounded retries.
JOB_UID=""
for _try in 1 2 3 4 5; do
    JOB_UID="$(kc -n "$NAMESPACE" get job "$JOB_NAME" -o jsonpath='{.metadata.uid}' 2> /dev/null || true)"
    [ -n "$JOB_UID" ] && break
    echo "submit: UID of job/$JOB_NAME not yet visible (attempt $_try/5) — retrying…" >&2
    sleep "$_try"
done
[ -n "$JOB_UID" ] || _abort_orphan "could not read UID of job/$JOB_NAME after 5 attempts (apiserver lag?)"

# Write the local run record before applying dependent resources.
_write_local_record "$JOB_UID"

# Apply the server stack with the bench Job as its owner.
echo "submit: apply server stack owned by job/$JOB_NAME (uid=$JOB_UID) → GC cascade teardown"
render_server "$JOB_UID" | kc apply -f - || _abort_orphan "server stack apply failed (bad node/quota/image?)"

# Persist a cluster-side run-id index for later status and collection.
echo "submit: apply run-id index configmap/llmb-submit-$RUN_ID (cold-reconnect resolver)"
render_index_cm | kc apply -f - || _abort_orphan "run-id index ConfigMap apply failed"

cat << EOF

submit: ✅ detached. The bench Job runs in-cluster on its SA token; the server is owned by it and will be
        GC-torn-down when the Job is GC'd. You may disconnect (laptop + tsh) now.

  run-id:  $RUN_ID
  status:  llmb-k8s status  $RUN_ID
  logs:    llmb-k8s logs    $RUN_ID
  cancel:  llmb-k8s cancel  $RUN_ID --cluster $PROFILE   # exact run only; asks for confirmation
  collect: llmb-k8s collect $RUN_ID
  watch:   llmb-k8s watch   $RUN_ID
  fleet:   llmb-k8s fleet --watch
EOF
