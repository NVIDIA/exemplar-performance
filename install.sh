#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

# Configuration
readonly MIN_PYTHON_VERSION="3.12"
readonly RECOMMENDED_PYTHON_VERSION="3.12"
# Optional uv version cap. Leave empty to install the latest uv and accept any installed version.
# Set (e.g. "0.9.28") when a recipe dependency breaks with newer uv releases.
readonly PINNED_UV_VERSION=""
readonly UV_VERSION_LABEL="${PINNED_UV_VERSION:-latest}"

# llmb-install applies the managed-python default to its own subprocesses.
case "${LLMB_DISABLE_MANAGED_PYTHON:-}" in
    1 | [Tt][Rr][Uu][Ee] | [Yy][Ee][Ss])
        echo "WARNING: LLMB_DISABLE_MANAGED_PYTHON is deprecated; set UV_NO_MANAGED_PYTHON=1 instead."
        unset UV_MANAGED_PYTHON UV_PYTHON_PREFERENCE LLMB_DISABLE_MANAGED_PYTHON
        export UV_NO_MANAGED_PYTHON=1
        ;;
esac

# Welcome banner
echo "════════════════════════════════════════════════════════════════════"
echo "🚀 Exemplar Performance - Quick Setup"
echo "════════════════════════════════════════════════════════════════════"
echo "This script will:"
echo "  • Install uv ($UV_VERSION_LABEL) if not already present"
echo "  • Set up a Python $RECOMMENDED_PYTHON_VERSION virtual environment (if needed)"
echo "  • Install essential tools (llmb-run, llmb-install)"
echo "  • Launch the main installer for benchmark configurations"
echo ""
echo "Note: Benchmark recipes require Python $RECOMMENDED_PYTHON_VERSION"
echo ""

# State for summary
UV_INSTALLED_NOW=false
UV_WAS_ADDED_TO_PATH=false
CREATED_VENV=false
VENV_DIR=""
MULTI_UV_WARNING_SHOWN=false
REPLACED_INCOMPATIBLE_ENV=false
SUMMARY_FILE=""
SUMMARY_INSTALL_PATH=""
SUMMARY_FAILED_WORKLOADS=""
SUMMARY_ASYNC_JOBS_SUBMITTED=""

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Prefer user-level uv install path when present.
ensure_local_uv_path() {
    local local_uv="$HOME/.local/bin/uv"
    if [[ -x $local_uv ]]; then
        case ":$PATH:" in
            *":$HOME/.local/bin:"*) ;;
            *)
                export PATH="$HOME/.local/bin:$PATH"
                UV_WAS_ADDED_TO_PATH=true
                ;;
        esac
    fi
}

# Get uv version string (e.g., 0.9.28)
get_uv_version() {
    uv --version 2> /dev/null | awk '{print $2}'
}

# Installer URL for the pinned uv version, or the latest release when unpinned.
uv_install_url() {
    if [[ -n $PINNED_UV_VERSION ]]; then
        echo "https://astral.sh/uv/${PINNED_UV_VERSION}/install.sh"
    else
        echo "https://astral.sh/uv/install.sh"
    fi
}

# Returns success (0) if version A is greater than version B.
version_gt() {
    [[ $1 =~ ^[0-9] ]] && [[ $2 =~ ^[0-9] ]] \
        && [[ $1 != "$2" ]] && [[ "$(printf '%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]
}

# Warn if multiple uv binaries are available in PATH.
# This warning is intentionally suppressed in low-risk cases to reduce noise.
warn_if_multiple_uv_installs() {
    if [[ $MULTI_UV_WARNING_SHOWN == true ]] || ! command_exists uv; then
        return 0
    fi

    local uv_paths
    uv_paths=$(type -ap uv 2> /dev/null | awk '!seen[$0]++')

    local count
    count=$(printf "%s\n" "$uv_paths" | sed '/^$/d' | wc -l | tr -d ' ')
    if [[ $count -gt 1 ]]; then
        local active_uv
        active_uv=$(command -v uv)
        local managed_uv="$HOME/.local/bin/uv"

        # Intentionally skip warning when managed uv is already first in PATH
        # and we did not need to modify PATH in this session.
        if [[ $active_uv == "$managed_uv" && $UV_WAS_ADDED_TO_PATH != true ]]; then
            MULTI_UV_WARNING_SHOWN=true
            return 0
        fi

        local active_version
        active_version=$(get_uv_version 2> /dev/null || echo "unknown")
        local version_note="v${active_version}"
        if [[ $active_uv == "$managed_uv" ]]; then
            version_note="v${active_version} ✅"
        fi

        echo ""
        echo "⚠️  Multiple uv binaries found in PATH:"
        while IFS= read -r uv_path; do
            [[ -n $uv_path ]] && echo "   - $uv_path"
        done <<< "$uv_paths"
        echo ""
        echo "   Active: $active_uv ($version_note)"
        echo ""
        echo "   If you run llmb-install in a new shell, a different uv may be"
        echo "   resolved. To avoid issues, ensure ~/.local/bin appears first"
        echo "   in your PATH (e.g., in ~/.bashrc)."
        echo ""
    fi

    MULTI_UV_WARNING_SHOWN=true
}

# Check if Python version meets minimum requirement
check_python_version() {
    local python_cmd="$1"
    # Validate MIN_PYTHON_VERSION format (e.g., "3.12")
    if [[ ! $MIN_PYTHON_VERSION =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "ERROR: Invalid MIN_PYTHON_VERSION format: $MIN_PYTHON_VERSION" >&2
        return 1
    fi
    local major="${MIN_PYTHON_VERSION%%.*}"
    local minor="${MIN_PYTHON_VERSION#*.}"
    if ! $python_cmd -c "import sys; exit(0 if sys.version_info >= ($major, $minor) else 1)" 2> /dev/null; then
        return 1
    fi
    return 0
}

# Get Python version string
get_python_version() {
    local python_cmd="$1"
    $python_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2> /dev/null || echo "unknown"
}

# Check if we're in a virtual environment (standard venv/virtualenv or conda)
in_virtual_env() {
    [[ -n ${VIRTUAL_ENV:-} ]] || [[ -n ${CONDA_DEFAULT_ENV:-} ]] || [[ -n ${CONDA_PREFIX:-} ]]
}

# Create a temporary summary file for llmb-install to write structured output.
init_summary_file() {
    local tmp_path=""
    tmp_path=$(mktemp "${TMPDIR:-/tmp}/llmb-install-summary.XXXXXX" 2> /dev/null || true)
    if [[ -n $tmp_path ]]; then
        SUMMARY_FILE="$tmp_path"
        export LLMB_INSTALL_SUMMARY_FILE="$SUMMARY_FILE"
    fi
}

# shellcheck disable=SC2317 # Invoked indirectly via trap
cleanup_summary_file() {
    if [[ -n $SUMMARY_FILE && -f $SUMMARY_FILE ]]; then
        rm -f "$SUMMARY_FILE"
    fi
}

# Parse summary file emitted by llmb-install (key=value format).
load_installer_summary() {
    local summary_path="$1"
    [[ -n $summary_path && -f $summary_path ]] || return 0

    while IFS= read -r line || [[ -n $line ]]; do
        line="${line%$'\r'}"
        [[ -z $line ]] && continue
        [[ $line == *=* ]] || continue

        local key="${line%%=*}"
        # ${line#*=} strips through the first '=' only, so values containing '=' are safe.
        local value="${line#*=}"

        case "$key" in
            version) ;;
            install_path)
                SUMMARY_INSTALL_PATH="$value"
                ;;
            failed_workloads)
                SUMMARY_FAILED_WORKLOADS="$value"
                ;;
            async_jobs_submitted)
                SUMMARY_ASYNC_JOBS_SUBMITTED="$value"
                ;;
        esac
    done < "$summary_path"
}

show_uv_install_error() {
    echo "❌ uv installation failed and uv is required."
    echo ""
    echo "   Install uv manually:"
    echo "     curl -LsSf $(uv_install_url) | sh"
    echo ""
    echo "   Re-run this script after installing uv."
    exit 1
}

# Install uv (pinned version when PINNED_UV_VERSION is set, otherwise latest)
install_uv() {
    if ! curl -LsSf "$(uv_install_url)" | sh; then
        echo "❌ uv installation failed"
        return 1
    fi

    ensure_local_uv_path
    hash -d uv 2> /dev/null || true

    # Check if uv is now available and matches the pinned version (if any)
    if command_exists uv; then
        local uv_version
        uv_version=$(get_uv_version || true)
        if [[ -n $uv_version && (-z $PINNED_UV_VERSION || $uv_version == "$PINNED_UV_VERSION") ]]; then
            echo "✅ uv $uv_version installed"
            UV_INSTALLED_NOW=true
            warn_if_multiple_uv_installs
            return 0
        fi
    fi

    echo "❌ uv installation failed: expected $UV_VERSION_LABEL, found $(get_uv_version || echo unknown)"
    return 1
}

# Ensure uv is compatible for this release.
# When pinned, we only correct versions newer than the known-compatible target.
ensure_compatible_uv() {
    ensure_local_uv_path

    if ! command_exists uv; then
        return 1
    fi

    local uv_version
    uv_version=$(get_uv_version || true)
    if [[ -z $uv_version ]]; then
        echo "⚠️  Unable to determine uv version. Installing uv ($UV_VERSION_LABEL)..."
        install_uv
        return $?
    fi

    if [[ -n $PINNED_UV_VERSION ]] && version_gt "$uv_version" "$PINNED_UV_VERSION"; then
        echo "⚠️  Detected uv $uv_version, but this release supports up to uv $PINNED_UV_VERSION."
        echo "   Installing uv $PINNED_UV_VERSION for compatibility..."
        install_uv
        return $?
    fi

    warn_if_multiple_uv_installs
    return 0
}

# Environment setup with proper sequencing
setup_environment() {
    echo "🔍 Checking Python environment..."

    # Step 1: Ensure uv is present (required)
    if command_exists uv; then
        if ! ensure_compatible_uv; then
            echo "❌ Cannot continue without a compatible uv installation."
            exit 1
        fi
    else
        echo "Installing uv ($UV_VERSION_LABEL, required)..."
        if ! install_uv; then
            show_uv_install_error
        fi
    fi

    # Step 2: Check if we're already in a good virtual environment
    local started_in_virtual_env=false
    if in_virtual_env; then
        started_in_virtual_env=true
    fi

    if [[ $started_in_virtual_env == true ]] && check_python_version python3; then
        local py_ver=$(get_python_version python3)
        echo "✅ Using existing virtual environment with Python $py_ver"
        return 0
    fi

    # Step 3: Need a new virtual environment
    if [[ $started_in_virtual_env == true ]]; then
        REPLACED_INCOMPATIBLE_ENV=true
        echo "⚠️  Active virtual environment is not compatible with recipe requirements."
        echo "   A new Python $RECOMMENDED_PYTHON_VERSION environment will be created for this install."
    fi

    echo "📦 Setting up new virtual environment..."
    echo "Using uv to create virtual environment with Python $RECOMMENDED_PYTHON_VERSION..."
    create_venv_with_uv
}

# Create venv with uv (can specify Python version)
create_venv_with_uv() {
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    VENV_DIR="$(realpath "$SCRIPT_DIR/../llmb_venv")"

    echo "Creating virtual environment with uv..."
    local -a python_pref=()
    if [[ -z ${UV_MANAGED_PYTHON:-}${UV_NO_MANAGED_PYTHON:-}${UV_PYTHON_PREFERENCE:-} ]]; then
        python_pref=(--managed-python)
    fi
    if ! uv venv --clear "${python_pref[@]}" -p "$RECOMMENDED_PYTHON_VERSION" "$VENV_DIR"; then
        echo "❌ Failed to create venv with uv"
        exit 1
    fi

    source "$VENV_DIR/bin/activate"
    uv pip install pip # Add pip for compatibility
    CREATED_VENV=true
    echo "✅ Virtual environment created and activated (Python $RECOMMENDED_PYTHON_VERSION)"
}

# Summarize environment and key next steps after running llmb-install
print_postinstall_summary() {
    local installer_status="$1"
    local status_label="FAILED"
    if [[ $installer_status -eq 0 ]]; then
        status_label="SUCCESS"
    fi

    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "📋 Install Summary: $status_label"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""

    if [[ $installer_status -eq 0 ]]; then
        echo "✅ Main installer completed."
    else
        echo "❌ Installation did not complete."
    fi
    echo ""

    if [[ -n $SUMMARY_INSTALL_PATH ]]; then
        echo "📁 Install Location (LLMB_INSTALL)"
        echo "   $SUMMARY_INSTALL_PATH"
        echo ""
    fi

    if [[ $installer_status -ne 0 && -n $SUMMARY_FAILED_WORKLOADS ]]; then
        local failed_display="${SUMMARY_FAILED_WORKLOADS//,/, }"
        echo "❌ Failed Workloads"
        echo "   $failed_display"
        echo ""
    fi

    # Show venv info if we created one
    if [[ $CREATED_VENV == true ]]; then
        if [[ $REPLACED_INCOMPATIBLE_ENV == true ]]; then
            echo "📦 New Virtual Environment Created (existing env incompatible)"
            echo "   Your active environment did not match Python $MIN_PYTHON_VERSION+ requirements."
        else
            echo "📦 Virtual Environment Created"
        fi
        echo "   $VENV_DIR"
        echo ""
        if [[ $installer_status -eq 0 ]]; then
            echo "   Activate this environment before running benchmarks:"
            echo "   source $VENV_DIR/bin/activate"
        else
            echo "   Faster retry (skip tool reinstall):"
            echo "   source $VENV_DIR/bin/activate"
            echo "   llmb-install"
        fi
        echo ""
    fi

    # Show PATH info only if we changed PATH for this session
    if [[ $UV_WAS_ADDED_TO_PATH == true ]]; then
        echo "⚡ uv Path Update"
        echo "   uv is available in this shell."
        echo "   To keep it available in new shells, add:"
        echo "   export PATH=\"$HOME/.local/bin:\$PATH\""
        echo "   (add to ~/.bashrc or ~/.zshrc)"
        echo ""
    elif [[ $UV_INSTALLED_NOW == true ]]; then
        echo "⚡ uv Installed"
        echo "   uv $(get_uv_version || echo "$UV_VERSION_LABEL") is installed and ready."
        echo ""
    fi

    if [[ $SUMMARY_ASYNC_JOBS_SUBMITTED == "true" ]]; then
        echo "⏳ Background Jobs Submitted"
        echo "   Some setup tasks are running as SLURM jobs."
        echo "   Check status with: squeue -u $USER"
        echo ""
    fi

    if [[ $installer_status -ne 0 ]]; then
        echo "────────────────────────────────────────────────────────────────────"
        echo "Address the error shown above, then retry:"
        echo "  ./install.sh"
        echo ""
    fi
}

# Main execution: Setup environment and validate tools
setup_environment

# Helper function to install a package
install_package() {
    local package_name="$1"
    local package_dir="$2"

    pushd "$package_dir" > /dev/null
    echo "  • Installing $package_name..."
    if ! uv pip install --quiet .; then
        echo ""
        echo "❌ Failed to install $package_name."
        echo "   If the error above is about finding a version of a dependency, check that"
        echo "   you can reach the package index it comes from. See ${package_dir%/}/README.md"
        echo "   for what that index requires."
        echo "   To install everything else without $package_name, remove or rename"
        echo "   ${package_dir%/} so this loop skips it, then rerun ./install.sh."
        exit 1
    fi
    popd > /dev/null
}

echo ""
echo "📦 Installing core tools..."

# Some checkouts carry packages under cli/ that a release does not, so install
# whatever this one contains rather than naming each package here. Each is
# installed from its own directory, which is also where uv picks up any package
# index that package declares for itself.
for package_dir in cli/*/; do
    [[ -f "${package_dir}pyproject.toml" ]] || continue
    install_package "$(basename "$package_dir")" "$package_dir"
done

echo "✅ Core tools installed successfully"

echo ""
echo "🚀 Launching main installer..."
echo ""

init_summary_file
# EXIT alone is sufficient: it fires on normal exit, explicit exit, and
# signal-induced termination, so no need for separate INT/HUP/TERM traps.
trap cleanup_summary_file EXIT

installer_status=0
if llmb-install "$@"; then
    installer_status=0
else
    installer_status=$?
fi

load_installer_summary "$SUMMARY_FILE"
if [[ $installer_status -ne 130 ]]; then
    print_postinstall_summary "$installer_status"
fi
exit "$installer_status"
