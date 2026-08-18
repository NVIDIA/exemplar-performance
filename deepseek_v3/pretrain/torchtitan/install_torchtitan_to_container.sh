#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Parameters
#SBATCH --exclusive
#SBATCH --job-name="deepseek_v3:torchtitan-container-setup"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00

set -euo pipefail

if [ "${BASH_VERSION:0:1}" -lt 4 ] || { [ "${BASH_VERSION:0:1}" -eq 4 ] && [ "${BASH_VERSION:2:1}" -lt 2 ]; }; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

# =============================================================================
# Configuration
# =============================================================================

GPU_TYPE="${GPU_TYPE:?GPU_TYPE is a required variable.}"

# Set architecture-specific config
case ${GPU_TYPE} in
    h100)
        DEEPEP_BRANCH="main"
        BUILD_AO="false"
        ;;
    gb200)
        DEEPEP_BRANCH="hybrid-ep"
        BUILD_AO="true"
        ;;
    b200)
        DEEPEP_BRANCH="main"
        BUILD_AO="true"
        ;;
    *)
        echo "Invalid GPU_TYPE: ${GPU_TYPE}. Use: h100, gb200, b200" >&2
        exit 1
        ;;
esac

# Pinned commits (tip of tree as of 2026-02-04)
DEEPEP_COMMIT_MAIN="567632dd59810d77b3cc05553df953cc0f779799"
DEEPEP_COMMIT_HYBRID="7febc6e25660af0f54d95dd781ecdcd62265ecca"

# Select DeepEP commit based on branch
if [[ ${DEEPEP_BRANCH} == "main" ]]; then
    DEEPEP_COMMIT="${DEEPEP_COMMIT_MAIN}"
else
    DEEPEP_COMMIT="${DEEPEP_COMMIT_HYBRID}"
fi

export LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL is a required variable.}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/pretrain_deepseek-v3-torchtitan
export TORCHTITAN_REPO="${LLMB_WORKLOAD}/torchtitan"

IMAGES_DIR="${LLMB_INSTALL}/images"
SQSH_FILE="${IMAGES_DIR}/nvidia+pytorch+25.12-py3.sqsh"
MODIFIED_SQSH="${SQSH_FILE}.modified"
BUILD_SCRIPT="${LLMB_WORKLOAD}/_container_build_internal.sh"

# Verify torchtitan exists
[[ -d ${TORCHTITAN_REPO} ]] || {
    echo "torchtitan not found at ${TORCHTITAN_REPO}"
    exit 1
}

# Verify container
[[ -f ${SQSH_FILE} ]] || {
    echo "Container not found: ${SQSH_FILE}"
    exit 1
}

# =============================================================================
# Generate internal build script
# =============================================================================

# Clean up any stale artifacts from previous failed runs
rm -f "${BUILD_SCRIPT}" "${MODIFIED_SQSH}"

cat > "${BUILD_SCRIPT}" << 'EOF'
#!/bin/bash
set -ex

# Config from env
GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
BUILD_AO=${BUILD_AO:-false}
DEEPEP_BRANCH=${DEEPEP_BRANCH:-main}
DEEPEP_COMMIT=${DEEPEP_COMMIT:?DEEPEP_COMMIT is a required variable.}
TORCHTITAN_REPO=${TORCHTITAN_REPO:?TORCHTITAN_REPO is a required variable.}

case ${GPU_TYPE} in
    h100)       export TORCH_CUDA_ARCH_LIST="9.0" ;;
    gb200|b200) export TORCH_CUDA_ARCH_LIST="10.0" ;;
    *)          echo "Unknown GPU_TYPE: ${GPU_TYPE}" >&2; exit 1 ;;
esac

echo "Build: GPU=${GPU_TYPE}, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, DeepEP=${DEEPEP_BRANCH}@${DEEPEP_COMMIT:0:8}"

# Install PyTorch stable (cu130); nightlies roll off the index — pin a GA triple
echo "Installing PyTorch 2.11+cu130 (stable)"
python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
python3 -m pip install --no-deps \
    torch==2.11.0+cu130 \
    torchvision==0.26.0+cu130 \
    torchaudio==2.11.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130

# System deps (RDMA libraries for both branches)
apt-get update && apt-get install -y --no-install-recommends \
    libibverbs-dev libibverbs1 librdmacm-dev librdmacm1 rdma-core ibverbs-utils

# Build rdma-core from source for HybridEP multinode (provides libmlx5)
if [[ "${DEEPEP_BRANCH}" == "hybrid-ep" ]]; then
    apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build libudev-dev libnl-3-dev libnl-route-3-dev pkg-config
    echo "Building rdma-core from source"
    RDMA_CORE_VERSION="v60.0"
    RDMA_BUILD_DIR="/tmp/rdma-core-build"
    rm -rf "${RDMA_BUILD_DIR}"
    mkdir -p "${RDMA_BUILD_DIR}"
    cd "${RDMA_BUILD_DIR}"
    git clone --depth 1 --branch "${RDMA_CORE_VERSION}" https://github.com/linux-rdma/rdma-core.git
    cd rdma-core
    bash build.sh
    export RDMA_CORE_HOME="${RDMA_BUILD_DIR}/rdma-core/build"
    export LD_LIBRARY_PATH="${RDMA_CORE_HOME}/lib:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="${RDMA_CORE_HOME}/lib:${LIBRARY_PATH:-}"
    echo "RDMA_CORE_HOME=${RDMA_CORE_HOME}"

    # hybrid_ep_cpp links with -l:libnvidia-ml.so.1, but the container runs with
    # NVIDIA_VISIBLE_DEVICES=void so driver libs aren't injected. Create a stub symlink.
    if [[ -d /usr/local/cuda/lib64/stubs ]] && [[ ! -f /usr/local/cuda/lib64/stubs/libnvidia-ml.so.1 ]]; then
        ln -sf libnvidia-ml.so /usr/local/cuda/lib64/stubs/libnvidia-ml.so.1
    fi
    export LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}"
fi

# NVSHMEM setup for DeepEP main branch (H100 and B200)
if [[ "${DEEPEP_BRANCH}" == "main" ]]; then
    echo "Setting up NVSHMEM"

    # Install NVSHMEM (required for DeepEP device linking with -rdc=true)
    # Pinned to the version shipped in the nvidia/pytorch:25.12-py3 base image
    python3 -m pip install nvidia-nvshmem-cu13==3.6.5

    # Download libcudacxx if needed
    if [[ ! -f "/usr/local/cuda/include/cuda/std/tuple" ]]; then
        cd /tmp
        wget -q https://github.com/NVIDIA/cccl/archive/refs/tags/v2.8.1.tar.gz -O cccl.tar.gz
        tar -xzf cccl.tar.gz
        cp -r cccl-2.8.1/libcudacxx/include/cuda/* /usr/local/include/cuda/ 2>/dev/null || \
            mkdir -p /usr/local/include/cuda && cp -r cccl-2.8.1/libcudacxx/include/cuda/* /usr/local/include/cuda/
        rm -rf cccl* /tmp/cccl*
        export CPLUS_INCLUDE_PATH="/usr/local/include:${CPLUS_INCLUDE_PATH:-}"
    fi
    
    # Find NVSHMEM
    NVSHMEM_DIR=$(python3 -c "import importlib.util; spec = importlib.util.find_spec('nvidia.nvshmem'); print(spec.submodule_search_locations[0] if spec else '')" 2>/dev/null || echo "")
    if [[ -n "${NVSHMEM_DIR}" && -f "${NVSHMEM_DIR}/include/nvshmem.h" ]]; then
        export NVSHMEM_DIR
        # Create symlinks
        cd "${NVSHMEM_DIR}/lib"
        for lib in libnvshmem_host.so.*; do [[ -f "$lib" ]] && ln -sf "$lib" libnvshmem_host.so 2>/dev/null || true; done
        cd - >/dev/null
        echo "NVSHMEM: ${NVSHMEM_DIR}"
    else
        echo "ERROR: NVSHMEM not found. DeepEP requires NVSHMEM for device linking (-rdc=true)." >&2
        echo "  Tried: find_spec('nvidia.nvshmem') → NVSHMEM_DIR='${NVSHMEM_DIR}'" >&2
        exit 1
    fi
fi

# Clone and build DeepEP
echo "Building DeepEP"
DEEPEP_DIR="/tmp/DeepEP"
git clone --branch "${DEEPEP_BRANCH}" https://github.com/deepseek-ai/DeepEP.git "${DEEPEP_DIR}"
cd "${DEEPEP_DIR}" && git checkout "${DEEPEP_COMMIT}"
if [[ "${DEEPEP_BRANCH}" == "hybrid-ep" ]]; then
    export HYBRID_EP_MULTINODE=1
fi
python3 -m pip install . --no-build-isolation
cd /tmp
rm -rf "${DEEPEP_DIR}"

# Install TorchAO if needed
if [[ "${BUILD_AO}" == "true" ]]; then
    echo "Installing TorchAO from source"
    TORCHAO_COMMIT="7bfc8da876a4360d4108e6882e0fda6ec4cb601d"
    TORCHAO_DIR="/tmp/torchao"
    git clone https://github.com/pytorch/ao.git "${TORCHAO_DIR}"
    cd "${TORCHAO_DIR}" && git checkout "${TORCHAO_COMMIT}"
    python3 -m pip install . --no-build-isolation
    cd /tmp
    rm -rf "${TORCHAO_DIR}"
fi

# Build TorchTitan
echo "Building TorchTitan"
cd "${TORCHTITAN_REPO}"
python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

# Verify
echo "Verification"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import deep_ep; print('DeepEP: OK')" || echo "DeepEP: installed (import needs GPU)"
python3 -c "import torchtitan; print('TorchTitan: OK')"
[[ "${BUILD_AO}" == "true" ]] && python3 -c "import torchao; print('TorchAO: OK')"

echo "Build Complete"
EOF

chmod +x "${BUILD_SCRIPT}"

# =============================================================================
# Run build in container
# =============================================================================

echo "Starting container build"

srun --account="${SLURM_ACCOUNT}" \
    --partition="${SLURM_PARTITION}" \
    --time=01:00:00 \
    --exclusive \
    --export="ALL,NVIDIA_VISIBLE_DEVICES=void,GPU_TYPE=${GPU_TYPE},\
BUILD_AO=${BUILD_AO},DEEPEP_BRANCH=${DEEPEP_BRANCH},\
DEEPEP_COMMIT=${DEEPEP_COMMIT},TORCHTITAN_REPO=${TORCHTITAN_REPO}" \
    --container-image="${SQSH_FILE}" \
    --container-mounts="${LLMB_WORKLOAD}:${LLMB_WORKLOAD}" \
    --container-writable \
    --container-save="${MODIFIED_SQSH}" \
    bash "${BUILD_SCRIPT}"

rm -f "${BUILD_SCRIPT}"

# Replace container
mv "${SQSH_FILE}" "${SQSH_FILE}.backup"
mv -f "${MODIFIED_SQSH}" "${SQSH_FILE}"

echo "Done! Container: ${SQSH_FILE}"
