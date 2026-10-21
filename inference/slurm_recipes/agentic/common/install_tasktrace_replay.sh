#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the `tasktrace` CLI (riptide-agent-replay) from source on the node
# running llmb-install and install it under $LLMB_WORKLOAD/bin/.
#
# Shared by the agentic recipes: each recipe's metadata.yaml lists this script
# as a setup task (`cmd: '../../common/install_tasktrace_replay.sh'`,
# job_type: local). llmb-install runs setup tasks with the recipe directory as
# the working directory and a per-workload $LLMB_WORKLOAD. The binary is built
# once per $LLMB_INSTALL, under tools/tasktrace/<target>/<source id>/, and
# copied into each workload's bin/, so recipes that share the tool reuse one
# build instead of compiling it again.
#
# Source of truth is the repository checkout: the replay tool lives in this
# tree at inference/tools/tasktrace (the vendored riptide-agent-replay,
# override TASKTRACE_SOURCE_DIR) and is built from there, so the tool version
# always matches the checked-out recipe. Nothing is cloned.
#
# Why a static musl binary, cross-compiled here: at replay time the binary is
# mounted read-only into the task container and executed there, so it must
# match the compute nodes' CPU architecture (which can differ from the login node's) and must
# not depend on the task image's glibc. The dependency graph is pure Rust, so
# rust-lld links the musl target for either architecture without a C cross toolchain
# and no compute-node allocation is needed. The artifact is verified to be a
# static ELF for the target machine. The result is cached by (source hash,
# target, requested toolchain) so re-running llmb-install is a no-op unless
# one of those changed.
#
# Framework contract (llmb-install, job_type: local):
#   $LLMB_INSTALL            installation dir (shared build + toolchain live here)
#   $LLMB_WORKLOAD           workload install dir (receives a copy of the binary)
#   $LLMB_NODE_ARCHITECTURE  compute nodes' CPU architecture from the cluster config
#                            (x86_64 | aarch64); selects the build target
#
# Login-node egress needed on the first install: static.rust-lang.org (rustup,
# toolchain, musl target std).
#
# Overrides:
#   TASKTRACE_SOURCE_DIR     in-tree source directory (contains Cargo.toml)
#   TASKTRACE_SHARED_DIR     where finished builds are kept
#                            (default: $LLMB_INSTALL/tools/tasktrace)
#   TASKTRACE_BUILD_DIR      rustup/cargo homes and the cargo target dir
#                            (default: $LLMB_INSTALL/.cache/tools/tasktrace)
#   TASKTRACE_TARGET_ARCH    x86_64 | aarch64 (default: $LLMB_NODE_ARCHITECTURE,
#                            else this node's `uname -m`, with a warning)
#   TASKTRACE_RUST_TOOLCHAIN rustup toolchain (default stable)
#   TASKTRACE_BUILD_JOBS     cargo build jobs (default: min(8, CPUs here))
#   TASKTRACE_SHOW_WARNINGS  set to 1 to show rustc lint warnings from the
#                            tasktrace crate (hidden by default; the package
#                            progress lines are always shown)

set -eu -o pipefail

export TT_TAG="[install_tasktrace_replay]" # consumed by the sourced library
# shellcheck source=tasktrace_install_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/tasktrace_install_lib.sh"
tt_require_bash
export LLMB_WORKLOAD=${LLMB_WORKLOAD:?LLMB_WORKLOAD not set (framework should provide).}
tt_locate_repo # COMMON_DIR, REPO_ROOT (setup tasks run inline, so BASH_SOURCE is this file)
TASKTRACE_SOURCE_DIR="${TASKTRACE_SOURCE_DIR:-$REPO_ROOT/inference/tools/tasktrace}"
TASKTRACE_RUST_TOOLCHAIN="${TASKTRACE_RUST_TOOLCHAIN:-stable}"
TASKTRACE_SHOW_WARNINGS="${TASKTRACE_SHOW_WARNINGS:-0}"
# The crate builds cleanly but carries a handful of lint warnings that add
# dozens of lines to the install log. Allow them unless asked to show them.
LINT_FLAGS="-A warnings"
if [[ $TASKTRACE_SHOW_WARNINGS == "1" ]]; then
    LINT_FLAGS=""
fi

HOST_ARCH="$(uname -m)"
TARGET_ARCH="${TASKTRACE_TARGET_ARCH:-${LLMB_NODE_ARCHITECTURE:-}}"
if [[ -z $TARGET_ARCH ]]; then
    TARGET_ARCH="$HOST_ARCH"
    echo "[install_tasktrace_replay] WARNING: LLMB_NODE_ARCHITECTURE is not set; building for this node's $HOST_ARCH." >&2
fi
case "$TARGET_ARCH" in
    aarch64 | arm64)
        TARGET_ARCH=aarch64
        TARGET=aarch64-unknown-linux-musl
        ;;
    x86_64 | amd64)
        TARGET_ARCH=x86_64
        TARGET=x86_64-unknown-linux-musl
        ;;
    *)
        echo "[install_tasktrace_replay] ERROR: unsupported target architecture '$TARGET_ARCH' (want x86_64 or aarch64)." >&2
        exit 1
        ;;
esac

BIN_DIR="$LLMB_WORKLOAD/bin"
VERSION_FILE="$BIN_DIR/tasktrace.version"
if [[ -n ${LLMB_INSTALL:-} ]]; then
    SHARED_DIR="${TASKTRACE_SHARED_DIR:-$LLMB_INSTALL/tools/tasktrace}"
    BUILD_DIR="${TASKTRACE_BUILD_DIR:-$LLMB_INSTALL/.cache/tools/tasktrace}"
else
    # Run outside llmb-install: keep everything under the workload directory.
    SHARED_DIR="${TASKTRACE_SHARED_DIR:-$LLMB_WORKLOAD/tasktrace-builds}"
    BUILD_DIR="${TASKTRACE_BUILD_DIR:-$LLMB_WORKLOAD}"
fi
RUST_DIR="$BUILD_DIR/rust"
mkdir -p "$LLMB_WORKLOAD" "$BIN_DIR"

# --- Locate the source ---------------------------------------------------------
SRC_DIR="$TASKTRACE_SOURCE_DIR"
[[ -f "$SRC_DIR/Cargo.toml" ]] || {
    echo "[install_tasktrace_replay] ERROR: no tasktrace source at $SRC_DIR (expected the vendored inference/tools/tasktrace; set TASKTRACE_SOURCE_DIR to override)." >&2
    exit 1
}
SOURCE_ID="$(tt_source_id "$SRC_DIR" target)"
echo "[install_tasktrace_replay] building from in-tree source $SRC_DIR"
echo "[install_tasktrace_replay] source id: $SOURCE_ID"

ARTIFACT_DIR="$SHARED_DIR/$TARGET/${SOURCE_ID//:/-}"
ARTIFACT="$ARTIFACT_DIR/tasktrace"
ARTIFACT_VERSION="$ARTIFACT_DIR/tasktrace.version"

matches() { tt_version_matches "$1" "source=$SOURCE_ID" "target=$TARGET" "toolchain=$TASKTRACE_RUST_TOOLCHAIN"; }
install_into_workload() {
    install -m 0755 "$ARTIFACT" "$BIN_DIR/tasktrace"
    cp "$ARTIFACT_VERSION" "$VERSION_FILE"
}

# --- Cache check: this workload first, then the build shared across workloads ---
if [[ -x $BIN_DIR/tasktrace ]] && matches "$VERSION_FILE"; then
    echo "[install_tasktrace_replay] up to date: $(tr '\n' ' ' < "$VERSION_FILE")"
    exit 0
fi
if [[ -x $ARTIFACT ]] && matches "$ARTIFACT_VERSION"; then
    echo "[install_tasktrace_replay] reusing the build shared across workloads: $ARTIFACT"
    install_into_workload
    echo "[install_tasktrace_replay] done: $BIN_DIR/tasktrace ($(tr '\n' ' ' < "$VERSION_FILE"))"
    exit 0
fi

# --- Build here (static musl, cross-compiled when the ISAs differ) --------------
echo "[install_tasktrace_replay] building on $(hostname) ($HOST_ARCH) for $TARGET"
export RUSTUP_HOME="$RUST_DIR/rustup" CARGO_HOME="$RUST_DIR/cargo"
export PATH="$CARGO_HOME/bin:$PATH"
if ! command -v cargo > /dev/null 2>&1; then
    echo "[install_tasktrace_replay] installing Rust toolchain ($TASKTRACE_RUST_TOOLCHAIN) under $RUST_DIR"
    mkdir -p "$RUST_DIR"
    tt_fetch_script https://sh.rustup.rs "$RUST_DIR/rustup-init.sh" rustup
    tt_run_installer rustup "$RUST_DIR/rustup-install.log" "$RUST_DIR/rustup-init.sh" \
        -y --profile minimal --default-toolchain "$TASKTRACE_RUST_TOOLCHAIN" --no-modify-path
fi
# Select the requested toolchain explicitly: cargo may already be on PATH with a
# different default, and the version file records this toolchain as the one used.
rustup toolchain install "$TASKTRACE_RUST_TOOLCHAIN" --profile minimal > /dev/null
rustup target add --toolchain "$TASKTRACE_RUST_TOOLCHAIN" "$TARGET" > /dev/null

CPUS_HERE=$(nproc 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 4)
JOBS="${TASKTRACE_BUILD_JOBS:-$((CPUS_HERE < 8 ? CPUS_HERE : 8))}"
cd "$SRC_DIR"
# Pure-Rust dependency graph: rust-lld links the musl target without a C cross toolchain.
upper=$(echo "$TARGET" | tr 'a-z-' 'A-Z_')
export "CARGO_TARGET_${upper}_LINKER=rust-lld"
export "CARGO_TARGET_${upper}_RUSTFLAGS=-C link-self-contained=yes -C target-feature=+crt-static $LINT_FLAGS"
export CARGO_TARGET_DIR="$BUILD_DIR/cargo-target"
cargo "+$TASKTRACE_RUST_TOOLCHAIN" build --release --locked --target "$TARGET" -j "$JOBS"
BUILT="$CARGO_TARGET_DIR/$TARGET/release/tasktrace"

# Verify the artifact is a static ELF for the target machine. It may not be
# runnable here (cross-compiled), so the helper inspects the headers instead
# of executing it.
python3 "$COMMON_DIR/tasktrace_recipe.py" elf-check --path "$BUILT" --arch "$TARGET_ARCH" --tag "[install_tasktrace_replay]"

mkdir -p "$ARTIFACT_DIR"
install -m 0755 "$BUILT" "$ARTIFACT"
if [[ $HOST_ARCH == "$TARGET_ARCH" ]]; then
    "$ARTIFACT" --help > /dev/null
fi
printf '%s\n' "source=$SOURCE_ID" "target=$TARGET" "toolchain=$TASKTRACE_RUST_TOOLCHAIN" "rustc=$(rustc "+$TASKTRACE_RUST_TOOLCHAIN" --version | awk '{print $2}')" \
    "built_on=$(hostname)" "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ARTIFACT_VERSION"
install_into_workload
echo "[install_tasktrace_replay] done: $BIN_DIR/tasktrace ($(tr '\n' ' ' < "$VERSION_FILE")) [shared build: $ARTIFACT]"
