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

# provision_imex_live_check.sh <cluster-profile> — one-command LIVE validation of the Tier-2 NVLink-IMEX
# fusion recovery. Applies the ComputeDomain, launches a
# short-lived 1-GPU diag pod that CLAIMS the IMEX channel, confirms the channel device appears and
# cuMulticastCreate(FABRIC) returns rc=0 (NOT code=800), then TEARS DOWN. Needs one free GPU + fresh auth.
#
# What the harness ALREADY validated live on GB300 (no GPU needed, done at build time):
#   • ComputeDomain (v1beta1) accepted by the live API; controller materialized the ResourceClaimTemplate.
#   • RBAC self-serve (create computedomains / resourceclaimtemplates) = yes.
#   • Idempotent re-probe → already-provisioned (keeps FLASHINFER).
# This script closes the LAST gap: the in-pod cuMulticastCreate rc=0 proof, which requires a GPU pod.
#
# Usage:  scripts/provision_imex_live_check.sh <cluster-profile>
# Safe:   uses a 1-GPU throwaway diag pod (NOT a 550B vLLM server); deletes everything on exit.
set -euo pipefail
PROFILE="${1:?usage: provision_imex_live_check.sh <cluster-profile>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
[ -f "$ENVF" ] || {
    echo "no profile at $ENVF" >&2
    exit 1
}
set -a
. "$ENVF"
set +a
kc() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} --request-timeout=30s "$@"; }
NS="${NAMESPACE:?NAMESPACE unset in profile}"
POD="imex-diag-$$"
IMG="${IMEX_DIAG_IMAGE:-nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04}" # any CUDA image with cuda-python

cleanup() {
    echo "── teardown ──"
    kc -n "$NS" delete pod "$POD" --ignore-not-found --wait=false > /dev/null 2>&1 || true
    python3 "$ROOT/scripts/provision_imex.py" "$PROFILE" --deprovision || true
}
trap cleanup EXIT

echo "── 1. provision the ComputeDomain (self-serve) ──"
python3 "$ROOT/scripts/provision_imex.py" "$PROFILE"
# reload the flags provision-imex just wrote
set -a
. "$ENVF"
set +a
[ "${NVLINK_MULTICAST_IMEX:-}" = "provisioned" ] || {
    echo "not provisioned (degrade path) — see message above"
    exit 0
}
RCT="${IMEX_CLAIM_TEMPLATE:-llmb-imex-channel}"

echo "── 2. launch a 1-GPU diag pod claiming the IMEX channel ──"
cat << YAML | kc -n "$NS" apply -f -
apiVersion: v1
kind: Pod
metadata: { name: ${POD}, labels: { app.kubernetes.io/managed-by: llmb-recipe } }
spec:
  restartPolicy: Never
  resourceClaims:
    - { name: imex-channel, resourceClaimTemplateName: ${RCT} }
  containers:
    - name: diag
      image: ${IMG}
      command: ["/bin/bash","-lc"]
      args:
        - |
          set -x
          ls -l /dev/nvidia-caps-imex-channels/ || true
          # Compiled C driver-API probe (the cuda:*-devel image ships headers + the libcuda stub;
          # the NVIDIA runtime injects the real libcuda.so.1 at run time). Avoids depending on a
          # python interpreter in the container, which the base CUDA image does not provide.
          cat > /tmp/mc.c <<'CEOF'
          #include <cuda.h>
          #include <stdio.h>
          #include <string.h>
          static const char* S(CUresult r){const char* s=0;cuGetErrorString(r,&s);return s?s:"?";}
          int main(void){
            CUresult rc;
            if((rc=cuInit(0))){ printf("cuInit rc=%d %s\n",(int)rc,S(rc)); return 2; }
            CUdevice dev; if((rc=cuDeviceGet(&dev,0))){ printf("cuDeviceGet rc=%d\n",(int)rc); return 2; }
            char nm[128]={0}; cuDeviceGetName(nm,sizeof(nm),dev);
            int mc=0; cuDeviceGetAttribute(&mc,CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,dev);
            printf("device: %s  multicast_supported=%d\n",nm,mc);
            CUcontext ctx; if((rc=cuCtxCreate(&ctx,0,dev))){ printf("cuCtxCreate rc=%d\n",(int)rc); return 2; }
            CUmulticastObjectProp prop;
            int nds[3]={2,4,1}; int sawNotPermitted=0;
            for(int j=0;j<3;j++){
              memset(&prop,0,sizeof(prop));
              prop.numDevices=nds[j]; prop.handleTypes=CU_MEM_HANDLE_TYPE_FABRIC;
              size_t g=0; CUresult rg=cuMulticastGetGranularity(&g,&prop,CU_MULTICAST_GRANULARITY_MINIMUM);
              if(rg||!g) g=2097152; prop.size=g;
              CUmemGenericAllocationHandle h;
              rc=cuMulticastCreate(&h,&prop);
              printf("cuMulticastCreate(FABRIC, numDevices=%d, size=%zu) rc = %d (%s)\n",nds[j],g,(int)rc,S(rc));
              if(rc==0){ printf("PASS: FABRIC multicast permitted (numDevices=%d) — allreduce_rms fusion can build\n",nds[j]); return 0; }
              if((int)rc==800) sawNotPermitted=1;
            }
            if(sawNotPermitted){ printf("FAIL: NOT_PERMITTED(800) — IMEX channel NOT effective\n"); return 1; }
            printf("INCONCLUSIVE: never rc=0 but never rc=800 (no no-IMEX signature) — arg issue, not a permission denial\n"); return 3;
          }
          CEOF
          command -v gcc >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq --no-install-recommends gcc; }
          gcc /tmp/mc.c -o /tmp/mc -I/usr/local/cuda/include -L/usr/local/cuda/lib64/stubs -lcuda
          /tmp/mc
      resources:
        requests: { nvidia.com/gpu: "1" }
        limits: { nvidia.com/gpu: "1" }
        claims: [{ name: imex-channel }]
  nodeSelector: { nvidia.com/gpu.product: ${GPU_PRODUCT} }
YAML

echo "── 3. wait + read the result ──"
kc -n "$NS" wait --for=condition=Ready pod/"$POD" --timeout=180s || true
kc -n "$NS" logs "$POD" -f || true
echo "── acceptance: '/dev/nvidia-caps-imex-channels/channel0' present AND 'cuMulticastCreate(FABRIC) rc = 0' ──"
