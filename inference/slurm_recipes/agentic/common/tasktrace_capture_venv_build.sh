#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the tasktrace-capture venv. Run by install_tasktrace_capture.sh either
# directly on the install host or on a compute node via srun; inputs arrive as
# TT_* environment variables (source dir, venv and uv dirs, versions, expected
# architecture, and TT_LIB, the shared helper library).

set -eu -o pipefail
export TT_TAG="[venv-build]" # consumed by the sourced library
# shellcheck source=tasktrace_install_lib.sh
source "$TT_LIB"
arch=$(uname -m)
if [[ $arch != "$TT_EXPECT_ARCH" ]]; then
    echo "[venv-build] ERROR: building on $arch but the venv must match $TT_EXPECT_ARCH." >&2
    exit 1
fi
echo "[venv-build] node: $(hostname)  arch: $arch  python: $TT_PYTHON"
# A private uv under the workload directory: independent of whatever the
# login or compute node has on PATH, and pinned for reproducibility.
export UV_INSTALL_DIR="$TT_UV_DIR/bin" UV_NO_MODIFY_PATH=1
export UV_CACHE_DIR="$TT_UV_DIR/cache" UV_PYTHON_INSTALL_DIR="$TT_UV_DIR/python"
export UV_LINK_MODE=copy
# Managed CPython only. llmb-install (>= 1.14) already exports UV_MANAGED_PYTHON=1
# to setup tasks, and uv rejects combining that with UV_PYTHON_PREFERENCE, so
# only set it when nothing has chosen a preference yet (e.g. a manual run).
if [[ -z ${UV_MANAGED_PYTHON:-} && -z ${UV_NO_MANAGED_PYTHON:-} && -z ${UV_PYTHON_PREFERENCE:-} ]]; then
    export UV_MANAGED_PYTHON=1
fi
export UV_PROJECT_ENVIRONMENT="$TT_VENV_DIR"
if [[ ! -x $TT_UV_DIR/bin/uv ]] || ! "$TT_UV_DIR/bin/uv" --version | grep -q " $TT_UV_VERSION\$"; then
    echo "[venv-build] installing uv $TT_UV_VERSION under $TT_UV_DIR"
    mkdir -p "$TT_UV_DIR"
    tt_fetch_script "https://astral.sh/uv/$TT_UV_VERSION/install.sh" "$TT_UV_DIR/uv-install.sh" uv
    tt_run_installer uv "$TT_UV_DIR/uv-install.log" "$TT_UV_DIR/uv-install.sh"
fi
export PATH="$TT_UV_DIR/bin:$PATH"
rm -rf "$TT_VENV_DIR"
cd "$TT_SRC_DIR"
uv python install "$TT_PYTHON"
uv sync --frozen --no-dev --python "$TT_PYTHON"
"$TT_VENV_DIR/bin/tasktrace-capture" --help > /dev/null
[[ -x $TT_VENV_DIR/bin/harbor ]] || {
    echo "[venv-build] harbor CLI missing from the venv" >&2
    exit 1
}
harbor_version=$("$TT_VENV_DIR/bin/python" -c 'import importlib.metadata as m; print(m.version("harbor"))')
python_version=$("$TT_VENV_DIR/bin/python" -c 'import platform; print(platform.python_version())')
printf '%s\n' "source=$TT_SOURCE_ID" "commit=$TT_SOURCE_COMMIT" "arch=$arch" "python_requested=$TT_PYTHON" \
    "python=$python_version" "uv=$TT_UV_VERSION" "harbor=$harbor_version" "built_on=$(hostname)" \
    "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TT_VENV_DIR.version"
echo "[venv-build] installed $TT_VENV_DIR (harbor $harbor_version, python $python_version)"
