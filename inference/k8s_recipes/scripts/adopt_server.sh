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

# adopt_server.sh <namespace> <cell-name> <owner-job-name> <owner-job-uid> [--dry-run]
# Add Job ownerReferences to this cell.s server Deployments so Kubernetes garbage collection
# releases them when the benchmark Job terminates. This runtime patch does not change recipe hashes.
set -euo pipefail

NS="${1:?usage: adopt_server.sh <namespace> <cell-name> <owner-job-name> <owner-job-uid> [--dry-run]}"
NAME="${2:?need the cell name (envelope.name)}"
JOB_NAME="${3:?need the owner Job name}"
JOB_UID="${4:-}"
DRY_RUN=0
[ "${5:-}" = "--dry-run" ] && DRY_RUN=1

# Use the profile-pinned context; KUBECTL may be replaced by tests.
kc() { "${KUBECTL:-kubectl}" ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }

# An empty owner UID is unsafe; leave the server unchanged and rely on existing cleanup backstops.
if [ -z "$JOB_UID" ]; then
    echo "adopt_server: WARN — empty owner uid for job/$JOB_NAME; leaving the server un-owned" \
        "(run.sh trap + governor orphan-sweep remain the backstops)." >&2
    exit 0
fi

# Match only this cell.s exact server, prefill, and decode Deployment names.
_deps="$(kc -n "$NS" get deploy -l app.kubernetes.io/managed-by=llmb-recipe \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' 2> /dev/null \
    | grep -E "^${NAME}-(server|prefill|decode)$" || true)"

if [ -z "$_deps" ]; then
    echo "adopt_server: no server Deployment for cell '$NAME' in ns/$NS — nothing to adopt (skipped)."
    exit 0
fi

_patch="{\"metadata\":{\"ownerReferences\":[{\"apiVersion\":\"batch/v1\",\"kind\":\"Job\",\"name\":\"$JOB_NAME\",\"uid\":\"$JOB_UID\",\"controller\":false,\"blockOwnerDeletion\":false}]}}"

while IFS= read -r _d; do
    [ -n "$_d" ] || continue
    if [ "$DRY_RUN" = 1 ]; then
        echo "adopt_server: [dry-run] would set ownerReference deploy/$_d -> job/$JOB_NAME (uid=$JOB_UID)"
        continue
    fi
    if kc -n "$NS" patch deploy "$_d" --type merge -p "$_patch" > /dev/null 2>&1; then
        echo "adopt_server: deploy/$_d now owned by job/$JOB_NAME — GC cascade-tears it down when the Job ends/deadlines"
    else
        echo "adopt_server: WARN — could not set ownerReference on deploy/$_d; leaving it un-owned" \
            "(run.sh trap + governor orphan-sweep remain the backstops)." >&2
    fi
done << EOF
$_deps
EOF
exit 0
