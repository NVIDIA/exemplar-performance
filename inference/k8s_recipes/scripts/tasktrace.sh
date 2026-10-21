#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck disable=SC1003,SC1090,SC2012,SC2015,SC2016,SC2034,SC2116,SC2207,SC2221,SC2222,SC2295,SC2317

# tasktrace.sh — the agentic lane's benchmark step, the analog of sweep.sh: applies a cell's rendered TaskTrace
# manifest, then runs one coordinator Job per rung, one after another; each spawns that rung's task pods.
#
# Usage:  scripts/tasktrace.sh <cell-dir> <cluster-profile> [run-id]
set -euo pipefail

CELL="${1:?usage: tasktrace.sh <cell-dir> <cluster-profile> [run-id]}"
PROFILE="${2:?need a cluster-profile name (cluster-profiles/<name>.env)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
MANIFEST="$CELL/rendered/tasktrace-job.yaml"
[ -f "$ENVF" ] || {
    echo "tasktrace.sh: no profile at $ENVF" >&2
    exit 1
}
[ -f "$MANIFEST" ] || {
    echo "tasktrace.sh: no $MANIFEST — run scripts/render.sh $CELL first" >&2
    exit 1
}
command -v envsubst > /dev/null || {
    echo "tasktrace.sh: envsubst not found (install gettext)" >&2
    exit 1
}

set -a
. "$ENVF"
set +a
# Resolve the cache PVC the way install did.
. "$ROOT/scripts/_model_cache.sh"
llmb::resolve_model_cache_pvc "$CELL" "$ENVF" || exit 1
kc() { "${KUBECTL:-kubectl}" ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }
. "$ROOT/scripts/_kubectl_resilient.sh"
. "$ROOT/scripts/_lib.sh"

read -r NAME RUNGS API_KEY_VAR < <(
    python3 - "$CELL/recipe.yaml" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
e, tt = r.get("envelope") or {}, r.get("tasktrace") or {}
rungs = ",".join(str(k) for k in ((tt.get("sweep") or {}).get("rungs") or [1]))
print(e.get("name", ""), rungs, ((tt.get("capture") or {}).get("api_key_secret") or "-"))
PY
)
[ -n "$NAME" ] || {
    echo "tasktrace.sh: could not read envelope.name from $CELL/recipe.yaml" >&2
    exit 1
}
MAX_RUNG="$(echo "$RUNGS" | tr ',' '\n' | sort -n | tail -1)"
# Size the run-id so the longest Job name, <cell>-tt<K>-<id>, fits 63 characters.
FIT_KIND="tt"
[ "$MAX_RUNG" -gt 1 ] && FIT_KIND="tt${MAX_RUNG}"
if [ -n "${3:-}" ]; then
    if [ "${LLMB_RUN_ID_FINAL:-0}" = 1 ]; then export RUN_ID="$3"; else
        export RUN_ID="$(python3 "$ROOT/scripts/run_id.py" "$CELL" --fit "$FIT_KIND" --label "$3" 2> /dev/null || echo "$3")"
    fi
else
    export RUN_ID="$(python3 "$ROOT/scripts/run_id.py" "$CELL" --fit "$FIT_KIND" 2> /dev/null || date -u +%Y%m%d-%H%M%S)"
fi
: "${OWNER:=}" "${TASKTRACE_NODE_SELECTOR:=}" # optional profile vars -> empty, not unset
export OWNER TASKTRACE_NODE_SELECTOR
if [ -z "$TASKTRACE_NODE_SELECTOR" ]; then
    echo "tasktrace: note: TASKTRACE_NODE_SELECTOR is empty — the Job lands on any node of the task image's architecture"
fi
if [ "$API_KEY_VAR" != "-" ]; then
    [ -n "${!API_KEY_VAR:-}" ] || {
        echo "tasktrace.sh: profile var $API_KEY_VAR (the Secret holding the endpoint API key) is empty in $ENVF" >&2
        exit 1
    }
    export "${API_KEY_VAR?}"
fi

if ! kc -n "$NAMESPACE" get pvc "${NAME}-artifacts" > /dev/null 2>&1; then
    echo "tasktrace: creating artifacts PVC ${NAME}-artifacts"
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

# Two Jobs of one cell on one node would share the measured cores. The run-owner Job also carries the cell
# label, so it is excluded.
ACTIVE="$(kc -n "$NAMESPACE" get jobs \
    -l "llmb.nvidia.com/cell=${NAME},app.kubernetes.io/component!=run-owner" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\n"}{end}' 2> /dev/null \
    | awk -F'\t' '($2+0)>0 {printf "%s ", $1}' || true)"
if [ -n "$ACTIVE" ]; then
    echo "tasktrace.sh: BLOCKED — an active TaskTrace Job for '$NAME' is already running: $ACTIVE" >&2
    echo "tasktrace.sh:   Two replays sharing a node's cores corrupt both cgroup measurements." >&2
    echo "tasktrace.sh:   Wait for it to finish, or follow its logs:" >&2
    echo "tasktrace.sh:     kubectl -n $NAMESPACE logs -f job/$ACTIVE" >&2
    exit 1
fi

# Capture BEFORE the Job is submitted; archive_run.py consumes this immutable receipt.
mkdir -p "$ROOT/results/$RUN_ID"
python3 "$ROOT/scripts/launch_attestation.py" "$CELL" "$RUN_ID" --out "$ROOT/results/$RUN_ID/launch_attestation.json"

# STRICT whitelist: ONLY cluster/runtime vars. The in-pod script's own $lower refs must survive envsubst.
WL='$NAMESPACE $RUN_ID $OWNER $IMAGE_PULL_SECRET $MODEL_CACHE_PVC $TASKTRACE_NODE_SELECTOR'
[ "$API_KEY_VAR" != "-" ] && WL="$WL \$$API_KEY_VAR"

# Per-cell objects first, then one coordinator Job per rung, one at a time.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
envsubst "$WL" < "$MANIFEST" > "$WORK/all.yaml"
python3 - "$WORK/all.yaml" "$WORK" << 'PY'
import sys, yaml
docs = [d for d in yaml.safe_load_all(open(sys.argv[1])) if d]
jobs = [d for d in docs if d.get("kind") == "Job"]
shared = [d for d in docs if d.get("kind") != "Job"]
if shared:
    with open(f"{sys.argv[2]}/shared.yaml", "w") as f:
        yaml.safe_dump_all(shared, f, sort_keys=False)
for d in jobs:
    rung = int((d.get("metadata") or {}).get("labels", {}).get("llmb.nvidia.com/rung", 1))
    with open(f"{sys.argv[2]}/rung-{rung:04d}.yaml", "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
print(f"tasktrace: {len(shared)} per-cell object(s) and {len(jobs)} rung coordinator Job(s); rungs run one at a time")
PY
if [ -s "$WORK/shared.yaml" ]; then
    kc apply -f "$WORK/shared.yaml"
fi
POD_READY_TIMEOUT="${TASKTRACE_POD_READY_TIMEOUT:-600s}" # the coordinator image is small; task pods wait inside it
overall=0
for doc in "$WORK"/rung-*.yaml; do
    RUNG="$(basename "$doc" .yaml | sed 's/^rung-0*//')"
    JOB_NAME="$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["metadata"]["name"])' "$doc")"
    echo "tasktrace: rung ${RUNG}: apply job/${JOB_NAME}  (cell=$NAME  run-id=$RUN_ID  ns=$NAMESPACE)"
    kc apply -f "$doc"
    # Bind the Job's lifetime to the run-owner when run.sh created one.
    if [ -n "${RUN_OWNER_UID:-}" ]; then
        bash "$ROOT/scripts/run_owner.sh" adopt-job "$NAMESPACE" "$JOB_NAME" "$RUN_OWNER_NAME" "$RUN_OWNER_UID" || true
    fi
    echo "tasktrace: following job/${JOB_NAME}  (Ctrl-C detaches; the Job keeps running)"
    COORD_SELECTOR="llmb.nvidia.com/run-id=${RUN_ID},llmb.nvidia.com/rung=${RUNG},llmb.nvidia.com/role=coordinator"
    if ! kc -n "$NAMESPACE" wait --for=condition=ready pod -l "$COORD_SELECTOR" --timeout="$POD_READY_TIMEOUT"; then
        # Not necessarily a failure: a short replay can already be Complete. Only complain if it is not terminal.
        if ! kc -n "$NAMESPACE" get "job/${JOB_NAME}" -o jsonpath='{.status.conditions[*].type}' 2> /dev/null | grep -qE 'Complete|Failed'; then
            echo "tasktrace: BLOCKED - pod for job/${JOB_NAME} did not become Ready within ${POD_READY_TIMEOUT}" >&2
            kc -n "$NAMESPACE" describe pod -l "$COORD_SELECTOR" >&2 || true
            exit 1
        fi
    fi
    kc -n "$NAMESPACE" logs -f "job/${JOB_NAME}" --all-containers --prefix 2> /dev/null || true
    set +e
    llmb::follow_job_to_terminal "$JOB_NAME" "$NAMESPACE" "${TASKTRACE_JOB_POLL_S:-10}"
    _jt=$?
    set -e
    case "$_jt" in
        0) echo "tasktrace: rung ${RUNG}: job/${JOB_NAME} complete" ;;
        1)
            echo "tasktrace: rung ${RUNG}: job/${JOB_NAME} FAILED (the coordinator exits non-zero unless every replay pod passes; see summary.json under /artifacts/${RUN_ID}/c${RUNG}/)" >&2
            overall=1
            ;;
        2)
            echo "tasktrace: rung ${RUNG}: job/${JOB_NAME} vanished after running (deleted/TTL-GC'd) — results are on the artifacts PVC; harvest with llmb-k8s collect" >&2
            overall=1
            ;;
    esac
done
exit "$overall"
