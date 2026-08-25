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

# apply-governor.sh — render (default) or apply the Phase-2 governor bundle for ONE namespace.
#
#   serving/governor/apply-governor.sh <cluster-profile> [--apply] [--dry-run-env]
#
# Renders, in order, to stdout:
#   1. ConfigMap  llmb-governor-script   (built from reconcile/governor_reconcile.sh via --from-file)
#   2. the envsubst-resolved templates/governor.yaml  (PVC + SA + Role + RoleBinding + CronJob)
#
# Default = RENDER ONLY (prints YAML, touches NO cluster) — safe offline. Pass --apply to `kubectl apply` it
# (the live PoC step, greenlit separately; depends on the Phase-1 control PVC substrate). Reads the RWX
# CONTROL_STORAGE_CLASS + optional knobs from the cluster profile, exactly like submit.sh reads its PVC vars.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" # inference/k8s_recipes
HERE="$ROOT/serving/governor"
SCRIPT="$HERE/reconcile/governor_reconcile.sh"
TEMPLATE="$HERE/templates/governor.yaml"

APPLY=0
DRY_RUN_ENV=0
PROFILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --apply)
            APPLY=1
            shift
            ;;
        --dry-run-env)
            DRY_RUN_ENV=1
            shift
            ;; # stamp DRY_RUN=1 into the CronJob (governor logs, mutates nothing)
        -h | --help)
            sed -n '2,16p' "$0"
            exit 0
            ;;
        *)
            PROFILE="$1"
            shift
            ;;
    esac
done
[ -n "$PROFILE" ] || {
    echo "usage: apply-governor.sh <cluster-profile> [--apply] [--dry-run-env]" >&2
    exit 2
}

ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
[ -f "$ENVF" ] || {
    echo "apply-governor: no profile at $ENVF" >&2
    exit 1
}
[ -f "$SCRIPT" ] || {
    echo "apply-governor: missing $SCRIPT" >&2
    exit 1
}
[ -f "$TEMPLATE" ] || {
    echo "apply-governor: missing $TEMPLATE" >&2
    exit 1
}

set -a
. "$ENVF"
set +a
: "${NAMESPACE:?profile must set NAMESPACE}"
: "${CONTROL_STORAGE_CLASS:?profile must set CONTROL_STORAGE_CLASS (an RWX storage class)}"

# Defaults for the optional knobs — applied HERE (not by envsubst, which does not honor ${VAR:-default}). A
# profile may override any of these; :=  assigns only when unset/empty so the template sees plain ${VAR}.
: "${CONTROL_SIZE:=5Gi}"
: "${GOVERNOR_IMAGE:=alpine/k8s:1.34.1}"
: "${GOVERNOR_SCHEDULE:=*/3 * * * *}"
: "${STALL_THRESHOLD:=1800}"
: "${HEARTBEAT_DEAD:=300}"
: "${STUCK_THRESHOLD:=900}"
: "${TIMEOUT_MULT:=2}"
: "${MIN_KILL_SECONDS:=3600}"
# ORPHAN_GRACE — never-adopted-orphan reap hold (the deploy->run window). 20m for the ENFORCE variant; the
# availableReplicas>=1 gate in sweep_orphans makes this SAFE even for 30-60min+ giant cold-loads (a loading
# server reports available=0 and is HELD until Available regardless of this grace). Backstopped by the
# per-server max-lifetime label ceiling (deploy.sh) for a server that never becomes Available.
: "${ORPHAN_GRACE:=1200}"
: "${DRY_RUN:=0}"
export CONTROL_SIZE GOVERNOR_IMAGE GOVERNOR_SCHEDULE STALL_THRESHOLD HEARTBEAT_DEAD \
    STUCK_THRESHOLD TIMEOUT_MULT MIN_KILL_SECONDS ORPHAN_GRACE DRY_RUN
[ "$DRY_RUN_ENV" = 1 ] && export DRY_RUN=1

kctl() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }

render() {
    # 1) the reconcile script as a ConfigMap (client-side render — no cluster needed).
    kubectl create configmap llmb-governor-script \
        --namespace "$NAMESPACE" \
        --from-file=governor_reconcile.sh="$SCRIPT" \
        --dry-run=client -o yaml
    # 2) the SA/Role/RoleBinding/PVC/CronJob bundle with cluster ${VARS} resolved.
    envsubst < "$TEMPLATE"
}

if [ "$APPLY" = 1 ]; then
    echo "apply-governor: applying governor bundle to ns=$NAMESPACE (control-sc=$CONTROL_STORAGE_CLASS)" >&2
    render | kctl apply -f -
else
    render
fi
