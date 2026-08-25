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

# deploy-observe.sh — render (default) or apply the Phase-2 governor SAFETY NET in OBSERVE / REPORT-ONLY mode
# for ONE namespace.
#
#   serving/governor/deploy-observe.sh <cluster-profile> [--apply] [--trigger]
#
# Renders, in order, to stdout:
#   1. ConfigMap  llmb-governor-script   (built from reconcile/governor_reconcile.sh via --from-file)
#   2. the envsubst-resolved templates/governor-observe.yaml  (SA + reduced Role + RoleBinding + CronJob)
#
# Default = RENDER ONLY (prints YAML, touches NO cluster). Pass --apply to `kubectl apply` it. Pass --trigger
# (implies --apply) to also kick ONE manual run immediately (`kubectl create job --from=cronjob/...`) so the
# observe report is produced without waiting a cycle.
#
# On-philosophy: vanilla k8s (plain CronJob + a namespace-scoped Role, NO CRD, NO cluster-admin). The observe
# Role grants ONLY get/list/watch on deployments/jobs/pods + get/create/update on the ONE report ConfigMap —
# NO delete/patch/scale — so it is structurally incapable of reaping. It mounts an emptyDir at /control (no
# Phase-1 RWX control PVC dependency), so it deploys even where Phase-1 is not wired.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" # inference/k8s_recipes
HERE="$ROOT/serving/governor"
SCRIPT="$HERE/reconcile/governor_reconcile.sh"
TEMPLATE="$HERE/templates/governor-observe.yaml"

APPLY=0
TRIGGER=0
PROFILE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --apply)
            APPLY=1
            shift
            ;;
        --trigger)
            APPLY=1
            TRIGGER=1
            shift
            ;;
        -h | --help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            PROFILE="$1"
            shift
            ;;
    esac
done
[ -n "$PROFILE" ] || {
    echo "usage: deploy-observe.sh <cluster-profile> [--apply] [--trigger]" >&2
    exit 2
}

ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
[ -f "$ENVF" ] || {
    echo "deploy-observe: no profile at $ENVF" >&2
    exit 1
}
[ -f "$SCRIPT" ] || {
    echo "deploy-observe: missing $SCRIPT" >&2
    exit 1
}
[ -f "$TEMPLATE" ] || {
    echo "deploy-observe: missing $TEMPLATE" >&2
    exit 1
}

set -a
. "$ENVF"
set +a
: "${NAMESPACE:?profile must set NAMESPACE}"
: "${IMAGE_PULL_SECRET:?profile must set IMAGE_PULL_SECRET}"

# Defaults for the optional knobs (envsubst does NOT honor ${VAR:-default}; := assigns only when unset/empty).
: "${GOVERNOR_IMAGE:=alpine/k8s:1.34.1}"
: "${GOVERNOR_OBSERVE_SCHEDULE:=*/15 * * * *}"
: "${OBSERVE_REPORT_CM:=governor-observe-report}"
: "${STALL_THRESHOLD:=1800}"
: "${HEARTBEAT_DEAD:=300}"
: "${STUCK_THRESHOLD:=900}"
: "${TIMEOUT_MULT:=2}"
: "${MIN_KILL_SECONDS:=3600}"
export NAMESPACE IMAGE_PULL_SECRET GOVERNOR_IMAGE GOVERNOR_OBSERVE_SCHEDULE OBSERVE_REPORT_CM \
    STALL_THRESHOLD HEARTBEAT_DEAD STUCK_THRESHOLD TIMEOUT_MULT MIN_KILL_SECONDS

kctl() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }

render() {
    kubectl create configmap llmb-governor-script \
        --namespace "$NAMESPACE" \
        --from-file=governor_reconcile.sh="$SCRIPT" \
        --dry-run=client -o yaml
    envsubst < "$TEMPLATE"
}

if [ "$APPLY" = 1 ]; then
    echo "deploy-observe: applying OBSERVE governor to ns=$NAMESPACE (image=$GOVERNOR_IMAGE, schedule='$GOVERNOR_OBSERVE_SCHEDULE')" >&2
    render | kctl apply -f -
    if [ "$TRIGGER" = 1 ]; then
        JOB="llmb-governor-observe-manual-$(date -u +%H%M%S)"
        echo "deploy-observe: triggering one manual run -> job/$JOB" >&2
        kctl -n "$NAMESPACE" create job "$JOB" --from=cronjob/llmb-governor-observe
        echo "$JOB"
    fi
else
    render
fi
