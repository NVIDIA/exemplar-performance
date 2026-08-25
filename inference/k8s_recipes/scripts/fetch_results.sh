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

# Pull benchmark artifacts off the (ReadWriteOnce) artifacts PVC.
#
# Because the artifacts PVC is RWO and the bench Job pod is gone by the
# time we want to extract data, we create a short-lived "mounter" Pod
# that mounts the PVC, stream the data out via `kubectl exec ... -- tar`
# per top-level entry, then delete the pod.
#
# Why per-entry streamed tar, not `kubectl cp` of the whole tree?
#   1. On Teleport-fronted kubectl, the websocket carrying `kubectl cp`
#      reliably closes abnormally (`close 1006`) for payloads above a
#      few GB. Per-entry streams keep each transfer small and let us
#      retry just the failed one if needed.
#   2. AIPerf materializes a full pre-tokenized `inputs.json` (~2.2 GB)
#      into every per-concurrency dir. We exclude that by default —
#      it's a debug artifact, not a result. Pass `--with-inputs` to
#      include it if you actually need it for replay.
#
# Usage:
#   ./scripts/fetch_results.sh <RUN_ID>                  # resolves cell/profile/PVC from results/.submits
#   ./scripts/fetch_results.sh --with-inputs <RUN_ID>
#   ./scripts/fetch_results.sh --partial <RUN_ID>      # best-effort recovery: grab whatever exists, exit 0
#   Re-running resumes verified entries; publish refuses an incomplete receipt.
#   ./scripts/fetch_results.sh --out /tmp/pull <RUN_ID>  # write here instead of recipe.env RESULTS_LOCAL_DIR
#   ./scripts/llmb-k8s collect 20260522-200000-my-namespace-12345 --cluster example-gpu-cluster
#   ./scripts/fetch_results.sh <CELL_DIR> <PROFILE> <RUN_ID>
#   ./scripts/fetch_results.sh --artifacts-pvc <PVC> <CELL_DIR> <PROFILE> <RUN_ID>
#   (a RESULTS_LOCAL_DIR you export also wins over recipe.env; --out takes precedence over both)

set -eu
# pipefail so a non-zero `kubectl exec ... -- tar c` isn't masked by a
# happy `tar xf -` (which would accept 0 bytes and exit 0).
set -o pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
. "$_SCRIPT_DIR/_lib.sh"
# Cert/auth-expiry resilience for the OPT-IN fetch path: re-auth proactively immediately before the (long)
# fetch, and heal + retry on any transient auth/network error mid-transfer, so a run that completes right at
# cert-expiry still fetches cleanly instead of false-failing. (Long/detached runs don't need this at all — the
# detached runs retain artifacts on the PVC for a later collect.)
# shellcheck source=_kubectl_resilient.sh
. "$_SCRIPT_DIR/_kubectl_resilient.sh"
# shellcheck source=_model_cache.sh
. "$_SCRIPT_DIR/_model_cache.sh"

llmb::require_cmd kubectl
llmb::require_cmd tar
llmb::require_cmd python3 # receipt evidence (files/bytes landed) is counted with python3

WITH_INPUTS=0
PARTIAL=0            # --partial: best-effort recovery — grab whatever exists, never exit non-zero
OUT_DIR=""           # optional local output directory
ARTIFACTS_PVC_ARG="" # exact durable run-record value; never infer it from an unrelated active profile
_RUN_NAMESPACE=""    # durable namespace for the one-argument reconnect path
# Capture a caller-exported RESULTS_LOCAL_DIR BEFORE load_env sources recipe.env (which would clobber it).
_CALLER_RLD="${RESULTS_LOCAL_DIR:-}"
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --with-inputs)
            WITH_INPUTS=1
            shift
            ;;
        --partial)
            PARTIAL=1
            shift
            ;;
        --out)
            OUT_DIR="${2:?--out needs a directory}"
            shift 2
            ;;
        --artifacts-pvc)
            ARTIFACTS_PVC_ARG="${2:?--artifacts-pvc needs a PVC name}"
            shift 2
            ;;
        -h | --help)
            sed -n '2,28p' "$0"
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*) die "Unknown flag: $1" ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

case "${#POSITIONAL[@]}" in
    1)
        RUN_ID_ARG="${POSITIONAL[0]}"
        # A bare run-id is convenient after `run`/`submit`, but recipe.env is not run identity: it may describe
        # a different cell/profile and therefore a different PVC. Resolve the durable submit record or fail
        # closed with the canonical cold-reconnect command. Never guess a claim for a fetch.
        _SUBMIT_RECORD="$RECIPE_ROOT/results/.submits/${RUN_ID_ARG}.json"
        [ -f "$_SUBMIT_RECORD" ] || die "No local submit record for run-id '$RUN_ID_ARG'. Use: llmb-k8s collect $RUN_ID_ARG --cluster <profile>, or pass <CELL_DIR> <PROFILE> <RUN_ID>."
        # macOS still ships Bash 3.2, which lacks the newer bulk line-reader builtin. Read each field independently so the
        # convenient bare-run-id path remains portable and empty values stay unambiguous.
        _run_field() {
            python3 - "$_SUBMIT_RECORD" "$1" << 'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(d.get(sys.argv[2]) or "")
PY
        }
        CELL_ARG="$(_run_field cell)"
        PROFILE_ARG="$(_run_field profile)"
        _RUN_NAMESPACE="$(_run_field namespace)"
        _RUN_ARTIFACTS_PVC="$(_run_field artifacts_pvc)"
        [ -n "$CELL_ARG" ] || die "Submit record $_SUBMIT_RECORD has no cell; pass <CELL_DIR> <PROFILE> <RUN_ID>."
        [ -n "$PROFILE_ARG" ] || die "Submit record $_SUBMIT_RECORD has no profile; pass <CELL_DIR> <PROFILE> <RUN_ID>."
        [ -n "$_RUN_NAMESPACE" ] || die "Submit record $_SUBMIT_RECORD has no namespace; pass <CELL_DIR> <PROFILE> <RUN_ID>."
        [ -n "$ARTIFACTS_PVC_ARG" ] || ARTIFACTS_PVC_ARG="${_RUN_ARTIFACTS_PVC:-}"
        ;;
    2)
        CELL_ARG="${POSITIONAL[0]}"
        PROFILE_ARG=""
        RUN_ID_ARG="${POSITIONAL[1]}"
        ;;
    3)
        CELL_ARG="${POSITIONAL[0]}"
        PROFILE_ARG="${POSITIONAL[1]}"
        RUN_ID_ARG="${POSITIONAL[2]}"
        ;;
    *)
        die "Usage: $0 [--with-inputs] [--artifacts-pvc <PVC>] [<CELL_DIR> [<PROFILE>]] <RUN_ID>"
        ;;
esac

# Load env without auto-generating RUN_ID (we want the one the caller passed).
[ -z "$PROFILE_ARG" ] || export CLUSTER="$PROFILE_ARG"
RUN_ID="$RUN_ID_ARG" llmb::load_env
# A profile is a connection default, not run identity. A detached run may have used a namespace override, so
# the durable submit record must win on the bare-run-id reconnect path just as its exact artifacts PVC does.
if [ -n "$_RUN_NAMESPACE" ]; then
    NAMESPACE="$_RUN_NAMESPACE"
    export NAMESPACE
fi

# The artifact claim uses the same cluster storage path as the model cache on the
# supported profiles. Some clusters can mount that storage only from the GPU
# pool; an unconstrained transient fetch pod can therefore attach the RWO claim
# to a CPU node and then fail its NFS mount. Reuse the canonical, fail-closed
# storage selector rather than inventing a second parser. The pod requests no
# GPU, and the tolerations only permit scheduling onto the selected pool.
llmb::model_cache_node_selector_yaml
FETCH_PLACEMENT_YAML=""
if [ -n "${MODEL_CACHE_NODE_SELECTOR_YAML:-}" ]; then
    FETCH_PLACEMENT_YAML="  nodeSelector: { ${MODEL_CACHE_NODE_SELECTOR_YAML} }
  tolerations:
  - { operator: Exists, effect: NoSchedule }
  - { operator: Exists, effect: PreferNoSchedule }"
fi

# The legacy path uses RECIPE_SHORTNAME from recipe.env. Declarative llm-perf
# cells use envelope.name as the per-cell artifacts PVC prefix, so derive it
# from the cell when provided.
if [ -n "$CELL_ARG" ]; then
    [ -f "$CELL_ARG/recipe.yaml" ] || die "Cell recipe not found: $CELL_ARG/recipe.yaml"
    CELL_NAME="$(sed -n 's/^  name: \(.*\)/\1/p' "$CELL_ARG/recipe.yaml" | head -1)"
    [ -n "$CELL_NAME" ] || die "Could not read envelope.name from $CELL_ARG/recipe.yaml"
    RECIPE_SHORTNAME="$CELL_NAME"
    export RECIPE_SHORTNAME
fi

ARTIFACTS_PVC_NAME="${ARTIFACTS_PVC_ARG:-${ARTIFACTS_PVC:-${RECIPE_SHORTNAME}-artifacts}}"
if ! printf '%s' "$ARTIFACTS_PVC_NAME" | grep -Eq '^[a-z0-9]([-a-z0-9]*[a-z0-9])?$' \
    || [ "${#ARTIFACTS_PVC_NAME}" -gt 253 ]; then
    die "Invalid artifacts PVC name: $ARTIFACTS_PVC_NAME"
fi

llmb::ensure_kube_context
# Refresh credentials before a potentially long artifact transfer when a hook is configured.
[ -n "$(llmb::reauth_cmd)" ] && { llmb::reauth > /dev/null 2>&1 || true; }

# Explicit --out wins, followed by a caller-exported RESULTS_LOCAL_DIR.
if [ -n "$OUT_DIR" ]; then
    RESULTS_LOCAL_DIR="$OUT_DIR"
elif [ -n "$_CALLER_RLD" ]; then
    RESULTS_LOCAL_DIR="$_CALLER_RLD"
fi

# Resolve results root: relative paths are anchored at RECIPE_ROOT.
if [ "${RESULTS_LOCAL_DIR#/}" = "$RESULTS_LOCAL_DIR" ]; then
    local_root="$RECIPE_ROOT/${RESULTS_LOCAL_DIR#./}"
else
    local_root="$RESULTS_LOCAL_DIR"
fi
local_dir="$local_root/$RUN_ID"
mkdir -p "$local_dir"

# Detached run.sh copies the pre-apply recipe hash into llmb-submit-$RUN_ID. Restore it before any archive/publish
# work on a different machine: this is evidence captured at launch, never a hash calculated while collecting.
_launch_receipt="$local_dir/launch_attestation.json"
if [ ! -f "$_launch_receipt" ]; then
    _launch_hash="$(llmb::kc -n "$NAMESPACE" get configmap "llmb-submit-$RUN_ID" \
        -o jsonpath='{.data.recipe_hash_at_launch}' 2> /dev/null || true)"
    _launch_captured="$(llmb::kc -n "$NAMESPACE" get configmap "llmb-submit-$RUN_ID" \
        -o jsonpath='{.data.recipe_hash_captured_at_utc}' 2> /dev/null || true)"
    if [ -n "$_launch_hash" ]; then
        python3 - "$_launch_receipt" "$RUN_ID" "${CELL_ARG:-}" "$_launch_hash" "$_launch_captured" << 'PY'
import json, re, sys
out, run_id, cell, recipe_hash, captured = sys.argv[1:]
if not re.fullmatch(r"[0-9a-f]{64}", recipe_hash):
    raise SystemExit("fetch: durable launch attestation has an invalid recipe_hash")
with open(out, "w") as fh:
    json.dump({"schema_version": 1, "kind": "recipe_hash_at_launch", "run_id": run_id,
               "cell": cell, "recipe_hash": recipe_hash,
               "captured_at_utc": captured or None}, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
        log "  provenance: restored launch-time recipe_hash receipt from durable submit index"
    fi
fi

# Per-entry markers make interrupted fetches resumable; the receipt records completeness.
DONE_DIR="$local_dir/.fetch_done"
mkdir -p "$DONE_DIR"
STAGE_DIR="$local_dir/.fetch_staging"
mkdir -p "$STAGE_DIR"
clear_abandoned_staging() {
    # A process killed mid-stream cannot run its EXIT trap and may leave a
    # private attempt directory behind. Fetches for one run already share one
    # helper-pod identity, so concurrent invocations are unsupported and a new
    # invocation owns this staging root. Keep the exact-path guard adjacent to
    # the recursive removal: an unresolved or broadened path must fail closed.
    case "$STAGE_DIR" in
        "$local_dir/.fetch_staging")
            for attempt_dir in "$STAGE_DIR"/*; do
                [ -d "$attempt_dir" ] || continue
                attempt_name="${attempt_dir##*/}"
                entry="${attempt_name%.*}"
                target="$local_dir/$entry"
                backup="$attempt_dir/.previous"
                # If the prior fetch died after parking the valid target but before
                # promoting the staged replacement, restore that target before
                # cleaning the abandoned attempt. A crash after promotion leaves the
                # new target in place, so the stale backup can then be discarded.
                if { [ -e "$backup" ] || [ -L "$backup" ]; } \
                    && { [ ! -e "$target" ] && [ ! -L "$target" ]; }; then
                    mv "$backup" "$target" || die "could not restore interrupted fetch backup for $entry"
                    log "  fetch recovery: restored previous complete entry $entry"
                fi
                rm -rf -- "$attempt_dir"
            done
            ;;
        *) die "unsafe fetch staging root: $STAGE_DIR" ;;
    esac
}
clear_abandoned_staging
MODE_TAG="$([ "$WITH_INPUTS" -eq 1 ] && echo withinputs || echo noinputs)"
RECEIPT="$local_dir/_fetch_status.json"
FETCH_SHORT=0 # set by write_receipt when the evidence contradicts a `complete: true` claim

# A complete receipt records and reconciles the files and bytes that landed locally.
# Count what actually landed locally, EXCLUDING the fetch's own bookkeeping (a receipt must never be able
# to vouch for itself). Prints "<files> <bytes>".
local_stats() {
    python3 - "$local_dir" << 'PY'
import os, sys
root = sys.argv[1]
n = b = 0
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames if d not in (".fetch_done", ".fetch_staging")]
    for f in filenames:
        if f == "_fetch_status.json":
            continue
        p = os.path.join(dirpath, f)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        n += 1
        b += st.st_size
print(n, b)
PY
}

# Count what the SOURCE holds, applying the same exclusion the transfer applies, so the two numbers are
# comparable. Best-effort: prints "" if the mounter can't be reached (recorded as remote_files: null →
# `reconciled: false`, which is enough for publish but NOT enough to authorise a delete).
remote_stats() {
    local excl=""
    [ "$WITH_INPUTS" -ne 1 ] && excl="! -name inputs.json"
    llmb::kc exec "$POD" -c mounter -- sh -c \
        "cd '$REMOTE_RUN_DIR' && find . \\( -type f -o -type l \\) $excl -exec ls -Lln {} + 2>/dev/null \
     | awk '{n++; s+=\$5} END {printf \"%d %d\\n\", n, s}'" 2> /dev/null | tail -1
}

# What ELSE is on this PVC that this fetch did not capture? The receipt vouches for /artifacts/$RUN_ID only,
# but auto-reclaim marks the WHOLE PVC. The audit found sibling directories of the run dir
# holding raw per-attempt records (30+ GB fleet-wide) that no fetch has ever pulled. Record the
# unfetched siblings so the reclaim reader can see exactly what a delete would destroy.
# Prints one "<name> <files>" line per non-empty sibling; empty output when there is nothing at risk.
sibling_stats() {
    llmb::kc exec "$POD" -c mounter -- sh -c \
        'cd /artifacts 2>/dev/null || exit 0; for e in * .[!.]*; do
       [ -e "$e" ] || continue; [ "$e" = "'"$RUN_ID"'" ] && continue
       n=$(find "$e" -type f 2>/dev/null | wc -l); [ "$n" -gt 0 ] && echo "$e $n"
     done' 2> /dev/null || true
}

# Write the fetch receipt consumed by publish.py / mark_reclaimable.py (reader: scripts/fetch_receipt.py).
# $1 = the caller's INTENT (true|false). `complete: true` additionally REQUIRES the evidence to back it up:
# files landed, bytes landed, and no shortfall against the source. Intent alone can only downgrade.
write_receipt() {
    local intent="$1" nfailed done_n failed_json="" e stats rstats
    local files_w bytes_w remote_f remote_b reconciled complete reason=""
    nfailed=$(printf '%s' "${FAILED:-}" | wc -w | tr -d ' ')
    done_n=$((${TOTAL:-0} - nfailed))
    for e in ${FAILED:-}; do failed_json="${failed_json}\"${e}\","; done
    failed_json="[${failed_json%,}]"

    stats="$(local_stats 2> /dev/null || echo '0 0')"
    files_w="${stats%% *}"
    bytes_w="${stats##* }"
    [ -n "$files_w" ] || files_w=0
    [ -n "$bytes_w" ] || bytes_w=0

    remote_f=null
    remote_b=null
    reconciled=false
    rstats="$(remote_stats || true)"
    case "$rstats" in
        [0-9]*\ [0-9]*)
            remote_f="${rstats%% *}"
            remote_b="${rstats##* }"
            reconciled=true
            ;;
    esac

    local sib_json="[]" sib_n=0 sib_raw
    sib_raw="$(sibling_stats || true)"
    if [ -n "$sib_raw" ]; then
        sib_json="$(printf '%s\n' "$sib_raw" | python3 -c '
import json, sys
out = []
for line in sys.stdin:
    parts = line.split()
    if len(parts) >= 2 and parts[-1].isdigit():
        out.append({"name": " ".join(parts[:-1]), "files": int(parts[-1])})
print(json.dumps(out))')"
        sib_n=$(printf '%s\n' "$sib_raw" | wc -l | tr -d ' ')
    fi

    complete="$intent"
    if [ "$complete" = true ]; then
        if [ "$files_w" -lt 1 ] || [ "$bytes_w" -lt 4096 ]; then
            complete=false
            reason="fetch wrote ${files_w} file(s) / ${bytes_w} B — nothing meaningful landed"
        elif [ "$reconciled" = true ] && [ "$files_w" -lt "$remote_f" ]; then
            complete=false
            reason="SHORT FETCH — ${files_w} of ${remote_f} source files landed ($((remote_f - files_w)) missing)"
        fi
    fi
    [ "$complete" = false ] && [ -z "$reason" ] && reason="entries failed to stream:${FAILED:- (see log)}"

    cat > "$RECEIPT" << JSON
{
  "receipt_version": 2,
  "run_id": "$RUN_ID",
  "remote_dir": "${REMOTE_RUN_DIR:-}",
  "with_inputs": $([ "$WITH_INPUTS" -eq 1 ] && echo true || echo false),
  "entries_total": ${TOTAL:-0},
  "entries_done": $done_n,
  "entries_resumed": ${RESUMED:-0},
  "failed": $failed_json,
  "files_written": $files_w,
  "bytes_written": $bytes_w,
  "remote_files": $remote_f,
  "remote_bytes": $remote_b,
  "reconciled": $reconciled,
  "pvc_unfetched": $sib_json,
  "complete": $complete,
  "incomplete_reason": $([ "$complete" = false ] && printf '"%s"' "$(printf '%s' "$reason" | tr -d '"\\\\')" || echo null),
  "updated_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

    if [ "$complete" = true ]; then
        log "  receipt: $files_w files / $bytes_w B landed$([ "$reconciled" = true ] && echo " (source: $remote_f files / $remote_b B — reconciled)" || echo " (source not countable — NOT reconciled)")"
    else
        err "  receipt: complete=false — $reason"
    fi
    if [ "$sib_n" -gt 0 ]; then
        log "  receipt: $sib_n other non-empty top-level entr$([ "$sib_n" = 1 ] && echo y || echo ies) on this PVC were NOT fetched"
        printf '%s\n' "$sib_raw" | sed 's/^/      /' >&2
        log "           → the PVC will NOT be auto-marked reclaimable (this fetch vouches for $RUN_ID only)"
    fi
    # Surface incomplete transfers and retain the source PVC for a safe retry.
    if [ "$intent" = true ] && [ "$complete" = false ]; then
        err ""
        err "  ✗ FETCH POSTCONDITION FAILED for $RUN_ID"
        err "    $reason"
        err "    Local tree at $local_dir is NOT a complete copy — do NOT delete the PVC."
        err "    Re-run: scripts/fetch_results.sh $RUN_ID   (resumes; re-streams anything missing)"
        FETCH_SHORT=1
    fi
}

POD="${RECIPE_SHORTNAME}-fetch-${RUN_ID}"

cleanup() {
    llmb::kc delete pod "$POD" --wait=false --ignore-not-found > /dev/null 2>&1 || true
}
trap cleanup EXIT

# Per-entry retry budget. Each entry retries up to MAX_TRIES with an
# exponential backoff. Failure on one entry doesn't abort the others —
# we collect failures and report at the end.
MAX_TRIES=3

log "Spawning transient mounter pod $POD ..."
# Disable client-side schema validation for this tiny, static Pod manifest.
# On Teleport-fronted clusters the OpenAPI schema download can timeout before
# the pod is submitted, even though the manifest itself is valid and the API is
# otherwise reachable. Server admission still validates the object.
cat << EOF | llmb::kc apply --validate=false -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${POD}
  labels:
    app.kubernetes.io/name: ${RECIPE_SHORTNAME}
    app.kubernetes.io/instance: ${RUN_ID}
    app.kubernetes.io/component: fetch
    app.kubernetes.io/managed-by: llmb-recipe
    llmb.nvidia.com/recipe: ${RECIPE_SHORTNAME}
    llmb.nvidia.com/run-id: ${RUN_ID}
    llmb.nvidia.com/owner: ${OWNER}
spec:
  restartPolicy: Never
${FETCH_PLACEMENT_YAML}
  containers:
  - name: mounter
    image: ${UTIL_IMAGE}
    command: ["sh", "-c", "sleep 3600"]
    volumeMounts:
    - name: artifacts
      mountPath: /artifacts
      readOnly: false
  volumes:
  - name: artifacts
    persistentVolumeClaim:
      claimName: ${ARTIFACTS_PVC_NAME}
      readOnly: false
EOF

log "Waiting for $POD to be Ready (timeout 5m)..."
# Retry the Ready wait across a transient auth/network blip: a single `kubectl wait` returns non-zero on an
# expired cert, which would abort the fetch. heal_auth re-auths on an auth signal; two short waits ride out a
# brief proxy stall without extending the effective ceiling much.
_pod_ready=0
for _wtry in 1 2 3; do
    if llmb::kc wait pod/"$POD" --for=condition=Ready --timeout=150s; then
        _pod_ready=1
        break
    fi
    llmb::heal_auth
done
if [ "$_pod_ready" -eq 0 ]; then
    err "Mounter pod failed to become Ready. Diagnostic:"
    llmb::kc describe pod "$POD" | tail -30 || true
    [ "$PARTIAL" = 1 ] && {
        err "(--partial) nothing to recover — mounter unavailable"
        exit 0
    }
    exit 1
fi

# Verify the remote run dir exists and enumerate its top-level entries.
REMOTE_RUN_DIR="/artifacts/$RUN_ID"
if ! llmb::kc exec "$POD" -c mounter -- test -d "$REMOTE_RUN_DIR" 2> /dev/null; then
    err "Remote dir not found on PVC: $REMOTE_RUN_DIR"
    err "Available runs on the PVC:"
    llmb::kc exec "$POD" -c mounter -- ls /artifacts 2>&1 || true
    [ "$PARTIAL" = 1 ] && {
        err "(--partial) nothing to recover for run $RUN_ID"
        exit 0
    }
    exit 1
fi

# Discover top-level entries: per-concurrency dirs, plus loose files like
# server_initial.id, and (after the symlink fix lands) a _shared/ dir.
# Sort numerically by concurrency where possible so the report ordering
# in the local copy matches the sweep order.
ENTRIES=$(llmb::kc exec "$POD" -c mounter -- sh -c "cd '$REMOTE_RUN_DIR' && ls -1 2>/dev/null" \
    | awk '
      /^concurrency_/ { c = $0; sub(/^concurrency_/, "", c); printf "%010d\t%s\n", c, $0; next }
      { printf "%010d\t%s\n", 9999999999, $0 }
    ' \
    | sort \
    | cut -f2)

if [ -z "$ENTRIES" ]; then
    err "Remote dir is empty: $REMOTE_RUN_DIR"
    [ "$PARTIAL" = 1 ] && {
        err "(--partial) nothing to recover"
        exit 0
    }
    exit 1
fi

log "Streaming $(echo "$ENTRIES" | wc -l | tr -d ' ') top-level entries from $REMOTE_RUN_DIR -> $local_dir"
[ "$WITH_INPUTS" -eq 1 ] && log "  (including inputs.json — use without --with-inputs to skip the ~2.2 GiB per-step dataset cache)"

# Build tar exclude args. By default we drop AIPerf's per-step
# inputs.json (~2.2 GiB each) and the deduplicated `_shared/inputs.json`
# (~2.2 GiB once per run, after the bench-job symlink fix). With
# `--with-inputs`, everything is included.
TAR_EXCLUDES=""
if [ "$WITH_INPUTS" -ne 1 ]; then
    TAR_EXCLUDES="--exclude=inputs.json"
fi

entry_stats() {
    python3 - "$1" << 'PY'
import os, sys
p = sys.argv[1]
n = b = 0
if os.path.lexists(p):
    if os.path.isdir(p) and not os.path.islink(p):
        for root, _dirs, files in os.walk(p):
            for name in files:
                try:
                    st = os.lstat(os.path.join(root, name))
                except OSError:
                    continue
                n += 1
                b += st.st_size
    else:
        st = os.lstat(p)
        n, b = 1, st.st_size
print(n, b)
PY
}

marker_matches() {
    local marker="$1" target="$2" mode="$3" stats
    [ -f "$marker" ] && { [ -e "$target" ] || [ -L "$target" ]; } || return 1
    stats="$(entry_stats "$target")" || return 1
    python3 - "$marker" "$mode" $stats << 'PY'
import json, sys
path, mode, files, nbytes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
try:
    marker = json.load(open(path))
except Exception:
    raise SystemExit(1)
ok = (marker.get("marker_version") == 2 and marker.get("mode") == mode
      and marker.get("files") == files and marker.get("bytes") == nbytes)
raise SystemExit(0 if ok else 1)
PY
}

write_done_marker() {
    local marker="$1" target="$2" mode="$3" stats tmp
    stats="$(entry_stats "$target")" || return 1
    tmp="${marker}.tmp.$$"
    python3 - "$tmp" "$mode" $stats << 'PY'
import json, sys
path, mode, files, nbytes = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with open(path, "w") as fh:
    json.dump({"marker_version": 2, "mode": mode, "files": files, "bytes": nbytes}, fh,
              sort_keys=True)
    fh.write("\n")
PY
    mv "$tmp" "$marker"
}

promote_staged_entry() {
    local attempt_dir="$1" entry="$2"
    local staged="$attempt_dir/$entry"
    local target="$local_dir/$entry" backup="$attempt_dir/.previous"
    case "$attempt_dir" in
        "$STAGE_DIR"/*) ;;
        *)
            err "unsafe fetch staging path: $attempt_dir"
            return 1
            ;;
    esac
    { [ -e "$staged" ] || [ -L "$staged" ]; } || {
        err "  staged entry missing after extraction: $entry"
        return 1
    }
    if [ -e "$target" ] || [ -L "$target" ]; then
        mv "$target" "$backup" || return 1
    fi
    if mv "$staged" "$target"; then
        rm -rf "$attempt_dir"
        return 0
    fi
    if [ -e "$backup" ] || [ -L "$backup" ]; then mv "$backup" "$target" || true; fi
    return 1
}

stream_entry() {
    local entry="$1"
    local stderr_file attempt_dir
    stderr_file="$(mktemp -t llmb-benchmark-fetch.XXXXXX)"
    # Defensive: $entry is interpolated into a single-quoted remote sh
    # command. Today the bench Job writes only well-controlled names
    # (`concurrency_<N>`, `_shared`, `server_initial.id`, `run_meta.json`),
    # but a regression there shouldn't translate into a remote-quoted
    # shell-injection sink here. Hard-allowlist names to printable
    # alphanumerics + `.`, `_`, `-`.
    case "$entry" in
        *[!A-Za-z0-9._-]* | "")
            err "  refusing to stream entry with non-allowlisted characters: $entry"
            rm -f "$stderr_file"
            return 1
            ;;
    esac
    attempt_dir="$(mktemp -d "$STAGE_DIR/${entry}.XXXXXX")" || {
        rm -f "$stderr_file"
        return 1
    }
    # -h on the cluster side so symlinks (the post-symlink-fix per-step
    # inputs.json) are streamed as their real contents — matters if
    # WITH_INPUTS=1.
    local rc=0
    llmb::kc exec "$POD" -c mounter -- sh -c \
        "cd '$REMOTE_RUN_DIR' && tar -ch $TAR_EXCLUDES -- '$entry'" \
        2> "$stderr_file" \
        | tar xf - -C "$attempt_dir" \
        || rc=$?
    if [ "$rc" -ne 0 ] && [ -s "$stderr_file" ]; then
        err "  stderr from failed stream ($entry):"
        sed 's/^/    /' "$stderr_file" | head -5 >&2
    fi
    rm -f "$stderr_file"
    if [ "$rc" -eq 0 ]; then
        promote_staged_entry "$attempt_dir" "$entry" || rc=$?
    fi
    if [ "$rc" -ne 0 ]; then
        case "$attempt_dir" in "$STAGE_DIR"/*) rm -rf "$attempt_dir" ;; esac
    fi
    return "$rc"
}

FAILED=""
TOTAL=$(echo "$ENTRIES" | wc -l | tr -d ' ')
DONE=0
RESUMED=0
for entry in $ENTRIES; do
    DONE=$((DONE + 1))
    # A completion marker is valid only while its corresponding local entry exists.
    marker="$DONE_DIR/${entry}.${MODE_TAG}"
    if marker_matches "$marker" "$local_dir/$entry" "$MODE_TAG"; then
        log "  [$DONE/$TOTAL] $entry — verified complete (resume), skipping"
        RESUMED=$((RESUMED + 1))
        continue
    fi
    if [ -f "$marker" ] || [ -e "$local_dir/$entry" ] || [ -L "$local_dir/$entry" ]; then
        err "  [$DONE/$TOTAL] $entry — completion evidence missing or stale; re-streaming atomically"
        rm -f "$marker"
    fi
    attempt=0
    while [ $attempt -lt $MAX_TRIES ]; do
        attempt=$((attempt + 1))
        if stream_entry "$entry"; then
            log "  [$DONE/$TOTAL] $entry (attempt $attempt) ✓"
            write_done_marker "$marker" "$local_dir/$entry" "$MODE_TAG"
            break
        fi
        if [ $attempt -lt $MAX_TRIES ]; then
            sleep_for=$((2 ** attempt))
            err "  $entry stream failed (attempt $attempt/$MAX_TRIES); retrying in ${sleep_for}s..."
            # A stream can fail because the kube cert expired mid-transfer — heal creds before the retry so the
            # next attempt reconnects cleanly instead of burning the retry budget on a dead session.
            llmb::heal_auth
            sleep "$sleep_for"
        else
            err "  $entry FAILED after $MAX_TRIES attempts"
            FAILED="$FAILED $entry"
        fi
    done
done

echo
if [ -n "$FAILED" ]; then
    write_receipt false # publish refuses an incomplete fetch receipt
    err "Some entries failed to stream:$FAILED"
    err "Inspect remote state with:"
    err "  kubectl -n $NAMESPACE exec -it $POD -c mounter -- ls -la $REMOTE_RUN_DIR"
    err "Re-run this script to retry (already-fetched entries are skipped); partial local copy is intact."
    [ "$PARTIAL" = 1 ] && {
        err "(--partial) keeping what was recovered at: $local_dir"
        exit 0
    }
    exit 1
fi

# ── cluster identity + live fingerprint (M3) ────────────────────────────────────────────────────────────
# Record cluster identity at collection time without overwriting an existing value.
if [ -f "$local_dir/run_meta.json" ]; then
    [ -n "${CLUSTER:-}" ] && python3 - "$local_dir/run_meta.json" "$CLUSTER" << 'PY'
import json, sys
p, c = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d["cluster"] = d.get("cluster") or c
json.dump(d, open(p, "w"), indent=2)
PY
    python3 "$_SCRIPT_DIR/cluster_fingerprint.py" \
        --context "${KUBE_CONTEXT:-$(kubectl config current-context 2> /dev/null)}" \
        --into "$local_dir/run_meta.json" > /dev/null 2>&1 \
        && log "  cluster: stamped ${CLUSTER:-?} + live fingerprint into run_meta" \
        || err "  cluster: fingerprint capture skipped (offline?) — cluster name stamped only"

    # The fetch deliberately enriches run_meta.json after its remote bytes have
    # landed. Refresh the local evidence marker so a later resume verifies the
    # enriched file instead of treating our own metadata stamp as corruption.
    _run_meta_marker="$DONE_DIR/run_meta.json.${MODE_TAG}"
    [ ! -f "$_run_meta_marker" ] \
        || write_done_marker "$_run_meta_marker" "$local_dir/run_meta.json" "$MODE_TAG"
fi

# ── PVC-handoff provenance (finding 2a) ─────────────────────────────────────────────────────────────────
# The per-entry `tar -ch` already carries the coord-dropped `.pvc_handoff` markers (RUN_ROOT/,
# ART/, and per-trial) since tar includes dotfiles. Belt-and-braces: if ANY handoff marker landed
# in the fetched tree, also drop one at the fetched-tree ROOT so a re-score pointed at the top of
# the download resolves handoff (0-byte model.patch = genuine no-op, not lost-in-transfer) even
# when only a sub-tree carrying the marker was pulled. NEVER fabricates one for a legacy exec run:
# it only copies a marker that the coord actually wrote.
_existing_marker="$(find "$local_dir" -name .pvc_handoff -type f 2> /dev/null | head -1 || true)"
if [ -n "$_existing_marker" ] && [ ! -f "$local_dir/.pvc_handoff" ]; then
    cp "$_existing_marker" "$local_dir/.pvc_handoff" 2> /dev/null \
        && log "  pvc-handoff: propagated durable marker to fetched-tree root (re-score resolves no-ops correctly)"
fi

# Assert the postcondition. write_receipt DOWNGRADES to complete=false if the evidence (files/bytes landed,
# reconciled against the source) doesn't back the claim — so this is a request, not a decree.
write_receipt true
if [ "$FETCH_SHORT" = 1 ]; then
    # A short fetch must fail LOUD. Nothing downstream may treat this tree as a complete copy, and the PVC
    # must stay unmarked (mark_reclaimable would refuse anyway — belt and braces: we don't even call it).
    [ "$PARTIAL" = 1 ] && {
        err "(--partial) keeping what was recovered at: $local_dir"
        exit 0
    }
    exit 1
fi

# Mark an artifacts PVC reclaimable only after a complete local fetch. The marker is advisory; deletion
# is a separate, human-gated `llmb-k8s reclaim --storage --apply`. Never fails a good fetch.
NAMESPACE="$NAMESPACE" KUBE_CONTEXT="${KUBE_CONTEXT:-}" \
    python3 "$_SCRIPT_DIR/mark_reclaimable.py" \
    --pvc "$ARTIFACTS_PVC_NAME" \
    --run-id "$RUN_ID" --local-dir "$local_dir" \
    ${LLMB_NO_RECLAIM:+--no-reclaim} || true

log "Done. Local results at: $local_dir"
ls -la "$local_dir" | head -40
echo
echo "Per-concurrency dirs:"
find "$local_dir" -mindepth 1 -maxdepth 1 -type d | sort
