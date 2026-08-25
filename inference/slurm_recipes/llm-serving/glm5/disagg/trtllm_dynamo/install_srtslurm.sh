#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Setup task 1/4 (job_type:srun, requires_gpus:true).
#
# Two things, both required before srtctl install can run on a compute node:
#
#   1. Clone github.com/NVIDIA/srt-slurm @ llmb/inf-beta to
#      $LLMB_WORKLOAD/srt-slurm  (login-node OK).
#
#   2. Create $SRT_SLURM_DIR/.venv ON AN AARCH64 COMPUTE NODE and editable-
#      install srtctl into it. Editable is REQUIRED by srt-slurm itself —
#      its package walks up its own source tree at runtime
#      (e.g. src/srtctl/cli/submit.py uses
#      Path(__file__).parent.parent.parent.parent to locate srtctl_root),
#      which only resolves correctly when the package is editable. srt-slurm's
#      own README prescribes `pip install -e .` for this reason.
#      We dispatch via `srun` because the login node is x86_64 and the
#      gb200/gb300 compute nodes are aarch64 (Grace). A venv built with the
#      login-node python has x86_64 binaries that fail "Exec format error"
#      when srtctl install's sbatch wrapper activates them on a compute node.
#
#      srtctl install (PR #148) defaults `--venv` to <srtctl_root>/.venv, so
#      placing the aarch64 venv there means install_model.sh does not need
#      to pass --venv at all.
#
# Why job_type:srun + requires_gpus:true rather than local:
#   We need SLURM_ACCOUNT + SLURM_PARTITION (gpu partition) injected by
#   augment_env_for_job_type so the inline `srun` below has the right
#   routing without any cluster-specific hardcoding. The framework runs
#   srun tasks INLINE on the install host — the actual aarch64 dispatch
#   is the explicit `srun` we issue below.

set -eu -o pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export LLMB_WORKLOAD=${LLMB_WORKLOAD:?LLMB_WORKLOAD not set (framework should provide).}
export SLURM_ACCOUNT=${SLURM_ACCOUNT:?SLURM_ACCOUNT not set.}
export SLURM_PARTITION=${SLURM_PARTITION:?SLURM_PARTITION not set.}

SRT_SLURM_REPO="https://github.com/NVIDIA/srt-slurm.git"
SRT_SLURM_BRANCH="llmb/inf-beta"
SRT_SLURM_DIR="$LLMB_WORKLOAD/srt-slurm"

mkdir -p "$LLMB_WORKLOAD"

# --- 1) clone -------------------------------------------------------------
if [[ -d "$SRT_SLURM_DIR/.git" ]]; then
    echo "[install_srtslurm] srt-slurm already cloned; fetching $SRT_SLURM_BRANCH"
    git -C "$SRT_SLURM_DIR" fetch --quiet origin
    git -C "$SRT_SLURM_DIR" checkout "$SRT_SLURM_BRANCH"
    git -C "$SRT_SLURM_DIR" pull --ff-only origin "$SRT_SLURM_BRANCH"
else
    echo "[install_srtslurm] cloning srt-slurm @ $SRT_SLURM_BRANCH → $SRT_SLURM_DIR"
    git clone --branch "$SRT_SLURM_BRANCH" --single-branch "$SRT_SLURM_REPO" "$SRT_SLURM_DIR"
fi

# --- 2a) login-side editable install into LLMB workload venv -------------
# install_model.sh runs as job_type:local on the login node (x86_64) and
# needs `srtctl` on PATH to issue the `srtctl install` command. The framework
# activates the LLMB workload venv ($VIRTUAL_ENV) before invoking us, so
# `uv pip install -e .` lands the (x86) srtctl wrapper there.
# Editable (-e) is required: srt-slurm walks up its own source tree at
# runtime to find sibling repo files (see header comment above for refs).
# A regular install resolves __file__ into site-packages and the walk-up
# breaks. srt-slurm's README prescribes -e for the same reason.
ACTIVE_ENV="${VIRTUAL_ENV:-${CONDA_PREFIX:-unknown}}"
echo "[install_srtslurm] uv pip install -e $SRT_SLURM_DIR (into LLMB workload env: $ACTIVE_ENV)"
if command -v uv > /dev/null 2>&1; then
    uv pip install -e "$SRT_SLURM_DIR"
else
    python -m pip install -e "$SRT_SLURM_DIR"
fi
if ! command -v srtctl > /dev/null 2>&1; then
    echo "[install_srtslurm] ERROR: srtctl not on PATH in LLMB workload venv after install." >&2
    exit 1
fi

# --- 2b) compute-side venv via srun --------------------------------------
# Block if a previous install_model job is still active. Re-running this
# task would rm -rf .venv, potentially invalidating the env the still-
# running install job depends on. install_model.sh writes this file on
# job submit and removes it once the job reaches a terminal state.
JOB_ID_FILE="$SRT_SLURM_DIR/.llmb-install-job-id"
if [[ -f $JOB_ID_FILE ]]; then
    ACTIVE_JOB_ID=$(< "$JOB_ID_FILE")
    if squeue -h -j "$ACTIVE_JOB_ID" 2> /dev/null | grep -q .; then
        echo "[install_srtslurm] ERROR: a previous install_model job is still active: $ACTIVE_JOB_ID" >&2
        echo "[install_srtslurm]        Wait for it to finish, or 'scancel $ACTIVE_JOB_ID', before rerunning llmb-install." >&2
        exit 1
    fi
    rm -f "$JOB_ID_FILE"
fi

# Always rebuild from scratch. A partial venv from a previously-failed
# install can leave a broken bin/srtctl in place that would falsely pass a
# "looks installed" check, masking the real failure.
echo "[install_srtslurm] building aarch64 .venv via srun ($SLURM_ACCOUNT, $SLURM_PARTITION)"
rm -rf "$SRT_SLURM_DIR/.venv"
# Intentionally NOT passing --gpus=N here. On clusters without GRES configured
# (if the cluster reports "No GPU GRES detected"), --gpus=1 errors out with
# "Invalid generic resource (gres) specification". On clusters that strictly
# require GPU requests, the convention is --gres=gpu:N, which is separate from
# --gpus and not impacted by omitting it. Omitting lets the partition apply
# its default allocation, which works in both worlds. We only need an aarch64
# compute node; we don't actually use the GPU for the pip install itself.
srun --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --time=00:15:00 --ntasks=1 --cpus-per-task=4 \
    bash -c "
        set -eu -o pipefail
        cd '$SRT_SLURM_DIR'

        # Find the right Python for this aarch64 compute node. srun inherits
        # the login node's env, which typically has x86_64 LLMB venvs on PATH;
        # a bare 'python3' would pick one of those and fail 'Exec format error'.
        # Default to /usr/bin/python3 (validated on gb200/gb300 Grace); override
        # via SRT_SLURM_PYTHON if your cluster keeps Python elsewhere — must
        # be an absolute path to a Python >= 3.10 that exists on the compute
        # node (not the login node).
        PY=\"\${SRT_SLURM_PYTHON:-/usr/bin/python3}\"
        [[ -x \$PY ]] || {
            echo \"[srun-venv] ERROR: \$PY not found on compute node.\" >&2
            echo \"[srun-venv]        Set SRT_SLURM_PYTHON to an absolute path to a Python >= 3.10 that exists on this compute node, then rerun llmb-install.\" >&2
            exit 1
        }

        echo \"[srun-venv] node: \$(hostname)  arch: \$(uname -m)  python3: \$PY\"

        # srt-slurm requires Python >= 3.10 (per its pyproject.toml). Fail
        # fast with a clear, actionable message rather than letting pip install
        # below die with a confusing dependency-resolution error.
        if ! \$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
            echo \"[srun-venv] ERROR: \$PY (\$(\$PY -V 2>&1)) is older than 3.10; srt-slurm requires >= 3.10.\" >&2
            echo \"[srun-venv]        Set SRT_SLURM_PYTHON to an absolute path to a Python >= 3.10 that exists on this compute node, then rerun llmb-install.\" >&2
            exit 1
        fi

        \$PY -m venv .venv
        source .venv/bin/activate
        pip install --quiet --upgrade pip
        pip install --quiet -e .
        echo \"[srun-venv] srtctl: \$(command -v srtctl)\"
     "

[[ -x "$SRT_SLURM_DIR/.venv/bin/srtctl" ]] || {
    echo "[install_srtslurm] ERROR: srtctl not in $SRT_SLURM_DIR/.venv/bin after build." >&2
    exit 1
}
echo "[install_srtslurm] done. compute-side srtctl: $SRT_SLURM_DIR/.venv/bin/srtctl"
