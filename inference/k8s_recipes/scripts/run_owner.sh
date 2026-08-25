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

# run_owner.sh manages the per-run Kubernetes ownership root. Server Deployments and benchmark
# Jobs become children of a watcher Job, so Kubernetes garbage collection releases resources after
# completion, failure, cancellation, or the configured deadline. Runtime ownership metadata is hash-neutral.
# The watcher releases only after an authoritative terminal Job state; unknown states keep resources.
#
# SUBCOMMANDS
#   ensure <ns> <cell> <run-id> <watched-job> [deadline_s] [ttl_s]
#         <watched-job> is a HINT ONLY (the caller's predicted Job name, for logs during the pre-adoption
#         window). The watcher decides on ownership; the name is never grounds for release.
#         Idempotently create the run-owner RBAC + the run-owner Job. Prints two eval-able lines to STDOUT:
#             RUN_OWNER_NAME=<cell>-runowner-<run-id>
#             RUN_OWNER_UID=<uid>
#         (all human logging goes to STDERR, so `eval "$(run_owner.sh ensure ...)"` is safe). On any failure
#         it still prints the two vars with an EMPTY uid, so the caller degrades to the legacy backstop.
#   adopt-job <ns> <job> <owner-name> <owner-uid>
#         Runtime-patch a bench Job's ownerReferences → the run-owner (post-apply; the Job is not
#         a GPU holder so a sub-second unowned window on IT is harmless — the SERVER is owned from birth).
#   adopt-deploy <ns> <cell> <owner-name> <owner-uid>
#         Runtime-patch the cell's EXACT server Deployment(s) (<cell>-server / -prefill / -decode) → the
#         run-owner. Used by the --skip-server path (server already exists; re-point it at THIS run's owner).
#   teardown <ns> <owner-name> [kube-context]
#         Delete the run-owner Job (background cascade) → k8s deletes the owned server + bench. Used by the
#         clean --teardown path and by run.sh's abort trap (Halt). Idempotent. Verifies the object was there
#         and is now gone — a not-found is reported as "nothing deleted, check your context", never as
#         success, because kc() otherwise only INHERITS KUBE_CONTEXT (pass [kube-context] standalone).
#
# BEST-EFFORT: every path exits 0 on cluster errors so the benchmark is never aborted by an ownership hiccup;
# the governor + adopt_server.sh remain the thin backstop for the residual (a genuine GC failure).
# Offline-testable: KUBECTL overrides the binary (a shim in selftest_run_owner.py); KUBE_CONTEXT is inherited.
set -euo pipefail

kc() { "${KUBECTL:-kubectl}" ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }

# alpine/k8s = sh + kubectl (same base the governor uses). Override per-cluster with RUNOWNER_IMAGE.
: "${RUNOWNER_IMAGE:=${GOVERNOR_IMAGE:-alpine/k8s:1.34.1}}"
# Run ceiling (hang cap). Defaults to the same 24h deploy.sh stamps as the server max-lifetime, so the owner
# deadline never clips a legitimate multi-hour cold-load + sweep; override per-cluster with RUNOWNER_DEADLINE_S.
: "${RUNOWNER_DEADLINE_S:=${SERVER_MAX_LIFETIME_S:-86400}}"
# Prompt post-terminal cleanup: how soon after the owner reaches terminal that GC cascades. Short by design.
: "${RUNOWNER_TTL_S:=120}"
# Watcher poll cadence (seconds). Small so fail/finish is detected quickly; get-only, negligible API load.
: "${RUNOWNER_POLL_S:=15}"

# ── owner-reference JSON (a non-... controller Job ref carrying the real owner uid) ──────────────────────────
_owner_patch() { # <owner-name> <owner-uid>
    printf '{"metadata":{"ownerReferences":[{"apiVersion":"batch/v1","kind":"Job","name":"%s","uid":"%s","controller":true,"blockOwnerDeletion":true}]}}' "$1" "$2"
}

cmd_ensure() {
    local ns="${1:?ensure needs <ns>}" cell="${2:?<cell>}" rid="${3:?<run-id>}" watched="${4:?<watched-job>}"
    local deadline="${5:-$RUNOWNER_DEADLINE_S}" ttl="${6:-$RUNOWNER_TTL_S}"
    local owner="${cell}-runowner-${rid}"

    # DNS-1123 label limit is 63 chars. Repro-clone cell names (already name-suffixed) push
    # "<cell>-runowner-<rid>" over 63. Rather than SKIP the run-owner — which drops the intrinsic
    # GPU-lifecycle guarantee to the slower adopt_server/governor backstop — COMPRESS the name to fit:
    # hash the full intended name (uniqueness, collision-safe across cells sharing a truncated prefix)
    # and truncate the descriptive cell prefix. Result stays valid DNS-1123: lowercase, ends on the hex
    # hash, no trailing/double hyphen.
    if [ "${#owner}" -gt 63 ]; then
        local h
        h="$(printf '%s' "$owner" | { shasum -a 256 2> /dev/null || sha256sum; } | cut -c1-8)"
        local suffix="-ro-${h}" # 12 chars, e.g. -ro-1a2b3c4d
        local prefix
        prefix="$(printf '%s' "$cell" | cut -c1-$((63 - ${#suffix})) | sed 's/-*$//')"
        owner="${prefix}${suffix}"
        echo "run_owner: owner name >63 chars — compressed to '$owner' (${#owner} chars) to preserve the run-owner." >&2
    fi

    # 1) RBAC (idempotent, one per namespace): the watcher only needs to READ Jobs to detect terminal state.
    if ! kc apply -f - >&2 2> /dev/null << YAML; then
apiVersion: v1
kind: ServiceAccount
metadata:
  name: llmb-runowner
  namespace: ${ns}
  labels: { app.kubernetes.io/managed-by: llmb-recipe, app.kubernetes.io/component: run-owner }
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: llmb-runowner
  namespace: ${ns}
  labels: { app.kubernetes.io/managed-by: llmb-recipe, app.kubernetes.io/component: run-owner }
rules:
  - { apiGroups: ["batch"], resources: ["jobs"], verbs: ["get", "list", "watch"] }
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: llmb-runowner
  namespace: ${ns}
  labels: { app.kubernetes.io/managed-by: llmb-recipe, app.kubernetes.io/component: run-owner }
subjects:
  - { kind: ServiceAccount, name: llmb-runowner, namespace: ${ns} }
roleRef: { kind: Role, name: llmb-runowner, apiGroup: rbac.authorization.k8s.io }
YAML
        echo "run_owner: WARN — could not apply run-owner RBAC in ns/$ns (backstop remains)." >&2
    fi

    # 2) the run-owner Job. activeDeadlineSeconds = hang ceiling; ttlSecondsAfterFinished = prompt cascade.
    #    The watcher waits for a Job THAT IT OWNS to reach a terminal state, then exits 0 → GC cascade.
    #
    #    THE TWO IMPLEMENTATIONS. This predicate is deliberately identical to run.sh's `_detach_scan_job`
    #    (558503a0): same jsonpath, same self-exclusion, same "one unambiguous reading"
    #    rule. It cannot be a shared FILE — this half runs as POSIX sh inside an alpine/k8s container with no
    #    repo mount, the other half is bash on the operator's laptop — so the shared thing is the CONTRACT, and
    #    selftest_run_owner.py cross-checks the two texts so they cannot drift apart silently.
    if ! kc apply -f - >&2 2> /dev/null << YAML; then
apiVersion: batch/v1
kind: Job
metadata:
  name: ${owner}
  namespace: ${ns}
  labels:
    app.kubernetes.io/managed-by: llmb-recipe
    app.kubernetes.io/component: run-owner
    llmb.nvidia.com/cell: ${cell}
    llmb.nvidia.com/run-id: ${rid}
spec:
  backoffLimit: 0
  completions: 1
  parallelism: 1
  activeDeadlineSeconds: ${deadline}
  ttlSecondsAfterFinished: ${ttl}
  template:
    metadata:
      labels: { app.kubernetes.io/managed-by: llmb-recipe, app.kubernetes.io/component: run-owner, llmb.nvidia.com/run-id: ${rid} }
    spec:
      restartPolicy: Never
      serviceAccountName: llmb-runowner
      # This is a control-plane watcher: it mounts no cache/artifact PVC, requests no GPU, and only talks
      # to the Kubernetes API. Do not inherit BENCH_NODE_SELECTOR here. A profile may intentionally pin the
      # benchmark to tainted GPU nodes, whose recipe-derived tolerations are not part of this generic Job;
      # copying that selector leaves the run-owner Pending and blocks the run before the server is deployed.
      containers:
        - name: watch
          image: ${RUNOWNER_IMAGE}
          imagePullPolicy: IfNotPresent
          command: ["/bin/sh", "-c"]
          env:
            - { name: NS,      value: "${ns}" }
            - { name: OWNER,   value: "${owner}" }
            - { name: WATCHED, value: "${watched}" }
            - { name: POLL,    value: "${RUNOWNER_POLL_S}" }
          args:
            - |
              set -eu
              # ── MY OWN UID ────────────────────────────────────────────────────────────────────────────
              # Children carry the JOB's uid in .ownerReferences, so the JOB's uid is what we must match.
              # The downward API cannot supply it: fieldRef exposes only this POD's name/uid/labels, and
              # ownerReferences is not a supported fieldPath — the pod's uid would match nothing. Reading
              # the pod and hopping to its ownerReference would need $(get pods) on top. So: SELF-LOOKUP by
              # our own Job name, which — unlike the watched Job's name — is not a prediction but this very
              # object's name, fixed at creation. Costs no extra RBAC ($(get jobs) is already granted).
              MYUID=""
              for _t in 1 2 3 4 5 6; do
                MYUID=\$(kubectl -n "\$NS" get job "\$OWNER" -o jsonpath='{.metadata.uid}' 2>/dev/null || true)
                if [ -n "\$MYUID" ]; then break; fi
                sleep "\${POLL:-15}"
              done
              if [ -z "\$MYUID" ]; then
                # RBAC missing / apiserver unreachable. DEGRADE TO HOLDING — never error out (a crash would
                # fail the owner and cascade-delete a healthy run) and never release blind.
                echo "run-owner: WARN — cannot read my own uid (job/\$OWNER in ns/\$NS; RBAC on batch/jobs?)."
                echo "run-owner: HOLDING until activeDeadlineSeconds. Never releasing on an unreadable cluster."
                # Idle in POLL-sized naps, not one long one: a stray $(sleep) inheriting this pod's
                # stdout must not outlive the container (or a test harness) by minutes.
                while :; do sleep "\${POLL:-15}"; done
              fi
              echo "run-owner: OWNER=job/\$OWNER uid=\$MYUID — releasing when a Job I OWN reaches a terminal state"
              echo "run-owner: (caller predicted job/\$WATCHED; that name is a HINT for logs only, never grounds to release)"

              JP='{range .items[*]}{.metadata.name}{"|"}{.metadata.ownerReferences[*].uid}'
              JP="\$JP"'{"|"}{.status.succeeded}{"|"}{.status.failed}{"\n"}{end}'
              PREV=""
              while :; do
                if ! RAW=\$(kubectl -n "\$NS" get jobs -o jsonpath="\$JP" 2>/dev/null); then
                  # A FAILED READ IS NOT "THE RUN FINISHED". Hold.
                  echo "run-owner: list failed (apiserver/authz) — HOLDING (a read error is never a terminal state)"
                  sleep "\${POLL:-15}"; continue
                fi
                OWNEDC=0; TERMC=0; TERMN=""; OWNEDN=""
                while IFS='|' read -r n uids s f; do
                  if [ -z "\$n" ]; then continue; fi
                  if [ "\$n" = "\$OWNER" ]; then continue; fi          # never my own object

                  case " \$uids " in *" \$MYUID "*) ;; *) continue ;; esac
                  OWNEDC=\$((OWNEDC+1)); OWNEDN="\$OWNEDN \$n"
                  if [ "\${s:-0}" -ge 1 ] 2>/dev/null || [ "\${f:-0}" -ge 1 ] 2>/dev/null; then
                    TERMC=\$((TERMC+1)); TERMN="\$n"
                  fi
                done <<INNER
              \$RAW
              INNER
                # THE ONE UNAMBIGUOUS READING: exactly one owned Job, and it is terminal.
                if [ "\$OWNEDC" -eq 1 ] && [ "\$TERMC" -eq 1 ]; then
                  echo "run-owner: job/\$TERMN (owned by me, uid=\$MYUID) reached a terminal state → releasing (GC cascade)"
                  exit 0
                fi
                if [ "\$OWNEDC" -gt 1 ]; then
                  STATE="ambiguous: \$OWNEDC Jobs carry my uid (\${OWNEDN# }) — HOLDING, refusing to guess which is the run"
                elif [ "\$OWNEDC" -eq 0 ]; then
                  STATE="no Job carries my uid yet (pre-adoption; caller predicted job/\$WATCHED) — HOLDING"
                else
                  STATE="job/\${OWNEDN# } is mine and still running — holding the GPU (as intended)"
                fi
                if [ "\$STATE" != "\$PREV" ]; then echo "run-owner: \$STATE"; PREV="\$STATE"; fi
                sleep "\${POLL:-15}"
              done
YAML
        echo "run_owner: WARN — could not create run-owner job/$owner (backstop remains)." >&2
        printf 'RUN_OWNER_NAME=\nRUN_OWNER_UID=\n'
        return 0
    fi

    # 3) read back the owner uid (retry — apply returns before the object is necessarily readable). A non-empty
    #    uid is what makes merge_run_owner.py / adopt-* actually stamp; empty → caller degrades to the backstop.
    local uid=""
    for _t in 1 2 3; do
        uid="$(kc -n "$ns" get job "$owner" -o jsonpath='{.metadata.uid}' 2> /dev/null || true)"
        [ -n "$uid" ] && break
        sleep "$_t"
    done
    if [ -n "$uid" ]; then
        echo "run_owner: run-owner job/$owner up (uid=$uid, deadline=${deadline}s, ttl=${ttl}s) — server+bench will GC-cascade on any terminal state" >&2
    else
        echo "run_owner: WARN — run-owner job/$owner created but uid unreadable; degrading to backstop." >&2
    fi
    printf 'RUN_OWNER_NAME=%s\nRUN_OWNER_UID=%s\n' "$owner" "$uid"
}

cmd_adopt_job() {
    local ns="${1:?adopt-job needs <ns>}" job="${2:?<job>}" owner="${3:?<owner-name>}" uid="${4:-}"
    if [ -z "$uid" ]; then
        echo "run_owner: adopt-job — empty owner uid; leaving job/$job un-owned (backstop remains)." >&2
        return 0
    fi
    if kc -n "$ns" patch job "$job" --type merge -p "$(_owner_patch "$owner" "$uid")" > /dev/null 2>&1; then
        echo "run_owner: job/$job now owned by run-owner job/$owner — GC cascades it when the run-owner terminates" >&2
    else
        echo "run_owner: WARN — could not set ownerReference on job/$job (backstop remains)." >&2
    fi
    return 0
}

cmd_adopt_deploy() {
    local ns="${1:?adopt-deploy needs <ns>}" cell="${2:?<cell>}" owner="${3:?<owner-name>}" uid="${4:-}"
    if [ -z "$uid" ]; then
        echo "run_owner: adopt-deploy — empty owner uid; leaving the server un-owned (backstop remains)." >&2
        return 0
    fi
    # EXACT server-object names only (never an open "^<cell>-" prefix, which would clobber a sibling cell's
    # server ownerRef — the …-1m vs …-1m-offload destructive-safety case, mirrored from adopt_server.sh).
    local deps
    deps="$(kc -n "$ns" get deploy -l app.kubernetes.io/managed-by=llmb-recipe \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2> /dev/null \
        | grep -E "^${cell}-(server|prefill|decode)$" || true)"
    if [ -z "$deps" ]; then
        echo "run_owner: adopt-deploy — no server Deployment for cell '$cell' in ns/$ns (skipped)." >&2
    fi
    while IFS= read -r d; do
        [ -n "$d" ] || continue
        if kc -n "$ns" patch deploy "$d" --type merge -p "$(_owner_patch "$owner" "$uid")" > /dev/null 2>&1; then
            echo "run_owner: deploy/$d now owned by run-owner job/$owner — GC cascades it when the run-owner terminates" >&2
        else
            echo "run_owner: WARN — could not set ownerReference on deploy/$d (backstop remains)." >&2
        fi
    done << EOF
$deps
EOF

    # THE SERVICE MUST MOVE WITH THE DEPLOYMENT. merge_run_owner.py stamps the cell's Service from birth too, so
    # on the --skip-server path that Service still names the PREVIOUS run's owner. If we re-point only the
    # Deployment, the old owner's terminal state GCs the Service out from under a LIVE server and the bench
    # loses its endpoint. Re-point both, with the same EXACT-name discipline (never a sibling …-1m-offload).
    # Service names across the lanes: <cell>-server (agg + disagg frontend), -prefill/-decode, -etcd/-nats.
    local svcs
    svcs="$(kc -n "$ns" get svc -l app.kubernetes.io/managed-by=llmb-recipe \
        -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2> /dev/null \
        | grep -E "^${cell}-(server|prefill|decode|etcd|nats)$" || true)"
    if [ -z "$svcs" ]; then
        echo "run_owner: adopt-deploy — no server Service for cell '$cell' in ns/$ns (skipped)." >&2
    fi
    while IFS= read -r s; do
        [ -n "$s" ] || continue
        if kc -n "$ns" patch svc "$s" --type merge -p "$(_owner_patch "$owner" "$uid")" > /dev/null 2>&1; then
            echo "run_owner: svc/$s now owned by run-owner job/$owner — GC cascades it when the run-owner terminates" >&2
        else
            echo "run_owner: WARN — could not set ownerReference on svc/$s (backstop remains)." >&2
        fi
    done << EOF
$svcs
EOF
    return 0
}

cmd_teardown() {
    local ns="${1:?teardown needs <ns>}" owner="${2:?<owner-name>}" ctx_arg="${3:-}"
    # CONTEXT (filed defect). kc() only INHERITS KUBE_CONTEXT, so a standalone operator invocation silently
    # targets whatever context happens to be current — and `--ignore-not-found` then makes that exit 0 with
    # "deleted successfully" while the GPU stays held on the real cluster. Two fixes: an explicit optional
    # <kube-context> argument, and — since we cannot always know the profile here — we no longer TRUST
    # --ignore-not-found: we verify the object is actually there, and actually gone.
    if [ -n "$ctx_arg" ]; then KUBE_CONTEXT="$ctx_arg"; fi
    local ctx="${KUBE_CONTEXT:-}"
    [ -n "$ctx" ] || ctx="$(kc config current-context 2> /dev/null || true)"
    local where="ns/$ns on context '${ctx:-<unknown>}'"

    if ! kc -n "$ns" get job "$owner" > /dev/null 2>&1; then
        # NOT "deleted successfully" — nothing was here to delete. Either it is already gone (fine) or we are
        # pointed at the WRONG CLUSTER and the GPU is still held somewhere else. Say which, loudly.
        echo "run_owner: teardown — run-owner job/$owner NOT FOUND in $where; NOTHING was deleted." >&2
        echo "run_owner:   If you expected it here, you are on the wrong context. Re-run with the right one:" >&2
        echo "run_owner:     scripts/run_owner.sh teardown $ns $owner <kube-context>" >&2
        return 0
    fi
    # Delete the run-owner → background cascade deletes the owned server + bench (delete, NOT scale-to-0, so no
    # idle-server 0/0 shell accumulates). Idempotent.
    if ! kc -n "$ns" delete job "$owner" --wait=false --ignore-not-found > /dev/null 2>&1; then
        echo "run_owner: WARN — could not delete run-owner job/$owner in $where (governor backstop remains)." >&2
        return 0
    fi
    # VERIFY, don't assume: gone, or at least tombstoned (deletionTimestamp set — --wait=false is async).
    local gone=0 _t
    for _t in 1 2 3; do
        if ! kc -n "$ns" get job "$owner" > /dev/null 2>&1; then
            gone=1
            break
        fi
        if [ -n "$(kc -n "$ns" get job "$owner" -o jsonpath='{.metadata.deletionTimestamp}' 2> /dev/null || true)" ]; then
            gone=1
            break
        fi
        sleep 1
    done
    if [ "$gone" = 1 ]; then
        echo "run_owner: deleted run-owner job/$owner in $where — GC is cascade-deleting the owned server + bench" >&2
    else
        echo "run_owner: WARN — delete of run-owner job/$owner in $where returned 0 but the object is still there;" >&2
        echo "run_owner:   the GPU may still be held (governor backstop remains). Check it by hand." >&2
    fi
    return 0
}

SUB="${1:?usage: run_owner.sh <ensure|adopt-job|adopt-deploy|teardown> ...}"
shift
case "$SUB" in
    ensure) cmd_ensure "$@" ;;
    adopt-job) cmd_adopt_job "$@" ;;
    adopt-deploy) cmd_adopt_deploy "$@" ;;
    teardown) cmd_teardown "$@" ;;
    *)
        echo "run_owner.sh: unknown subcommand '$SUB' (ensure|adopt-job|adopt-deploy|teardown)" >&2
        exit 2
        ;;
esac
