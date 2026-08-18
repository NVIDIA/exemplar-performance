#!/bin/bash
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

#SBATCH --job-name="system_info:microbenchmark"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:15:00

set -u -o pipefail

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=microbenchmark
export WORKLOAD=system_info
export FW_VERSION=26.06.01

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

# GPU type (e.g. gb200, gb300, b200, h100), provided by llmb-run. Lower-cased
# in place for matching (mirrors the cpu_overhead recipe). Used to gate
# platform-specific expectations such as the NVL72 NVLink count and IMEX.
# Optional here: the script still runs if it is unset.
export GPU_TYPE="${GPU_TYPE:-}"
GPU_TYPE="${GPU_TYPE,,}"

FAILED_STEPS=0

print_banner() {
    local title="$1"
    echo
    echo "================================================================================"
    echo "${title}"
    echo "================================================================================"
}

run_step() {
    local step_id="$1"
    local title="$2"
    local cmd="$3"
    local rc=0

    print_banner "[${step_id}] ${title}"

    bash -o pipefail -c "${cmd}"
    rc=$?

    if [[ ${rc} -ne 0 ]]; then
        echo
        echo "[FAILED] exit code ${rc} (non-fatal, continuing)"
        FAILED_STEPS=$((FAILED_STEPS + 1))
    fi
}

print_banner "System Info Collection Start"
echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"
echo "LLMB_EXPERIMENT_DIR: ${LLMB_EXPERIMENT_DIR:-unset}"
echo "Timestamp: $(date -Iseconds)"

run_step "1a" "lscpu - CPU details, SKU, core count" "lscpu"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_cpu_governor() {
    if ! compgen -G '/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor' > /dev/null; then
        echo "cpufreq not available on this system (no scaling_governor files)"
        return
    fi

    local govs
    govs=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor)
    # Count per distinct governor: stays compact on high-core-count systems and
    # still reveals any split when cores disagree.
    printf '%s\n' "${govs}" | sort | uniq -c
    echo

    if [ "$(printf '%s\n' "${govs}" | sort -u)" = "performance" ]; then
        echo "[OK] All cores using performance governor."
    else
        echo "[WARNING] One or more cores not using performance governor."
        echo "          'performance' is recommended for benchmarking to avoid frequency scaling noise."
    fi
}

run_step "1b" "cpufreq scaling_governor - CPU frequency governor per core" \
    "$(declare -f check_cpu_governor); check_cpu_governor"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_cpu_cstates() {
    local d=/sys/devices/system/cpu/cpu0/cpuidle
    if [ ! -d "${d}" ] || ! compgen -G "${d}/state*" > /dev/null; then
        echo "cpuidle not available on this system (no states under ${d})"
        return
    fi

    local count=0 disabled=0 f
    for f in "${d}"/state*; do
        [ -d "${f}" ] || continue
        count=$((count + 1))
        printf '  %s: %s (disable=%s)\n' "$(basename "${f}")" \
            "$(cat "${f}/name" 2> /dev/null)" "$(cat "${f}/disable" 2> /dev/null)"
        [ "$(cat "${f}/disable" 2> /dev/null)" = "1" ] && disabled=$((disabled + 1))
    done
    echo

    if [ "${count}" -le 2 ]; then
        echo "[WARNING] Only ${count} CPU idle state(s) exposed on cpu0."
        echo "          Deep C-states may be disabled in firmware; active cores may not reach expected boost clocks."
    elif [ "${disabled}" -gt 0 ]; then
        echo "[WARNING] ${disabled} of ${count} idle states disabled on cpu0 (disable=1)."
    else
        echo "[OK] ${count} CPU idle states exposed on cpu0, none disabled."
    fi
}

run_step "1c" "cpuidle - CPU C-state availability (boost-clock readiness)" \
    "$(declare -f check_cpu_cstates); check_cpu_cstates"
run_step "2" "lspci -v - networking hardware and software layers" "lspci -v"
run_step "3" "numactl -H - NUMA node assignments" "numactl -H"
run_step "4" "cat /proc/cmdline - Linux kernel arguments" "cat /proc/cmdline"
run_step "5" "systemd-detect-virt - virtualization check" \
    "systemd-detect-virt || true"
run_step "6" "getconf PAGE_SIZE - memory page size" "getconf PAGE_SIZE"
run_step "7" "dmesg | grep -i smmu - SMMU CMDQV feature status" \
    "dmesg | grep -i smmu || echo 'No SMMU lines found or dmesg access is restricted.'"
run_step "8a" "nvidia-smi -q - GPU hardware, firmware, software details" "nvidia-smi -q"
run_step "8b" "nvidia-smi topo -m - GPU/NIC topology matrix (NVLink, PCIe, NUMA affinity)" \
    "nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g'"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_gpu_ecc_remap() {
    command -v nvidia-smi > /dev/null 2>&1 || {
        echo "nvidia-smi not found"
        return 127
    }

    # nvidia-smi --query-gpu field availability is driver-gated: an older
    # nvidia-smi rejects the whole query at the first selector it doesn't know
    # ("not a valid field to query") rather than returning N/A. ECC and remap
    # are queried separately so an unsupported remap field can't take ECC down
    # with it, and each query degrades (via exit code) instead of failing the
    # step. Row remapping is an Ampere+ feature, so the remap fields are present
    # on H100/GB200 with a current driver; on an older driver the remap query is
    # skipped (not failed). All four remap values come from one NVML call, so the
    # count and boolean fields are gated together -- no benefit to dropping one.
    local warn=0

    echo "--- uncorrected ECC (aggregate) ---"
    local ecc ecc_rc
    ecc=$(nvidia-smi --query-gpu=pci.bus_id,name,ecc.errors.uncorrected.aggregate.total --format=csv 2>&1)
    ecc_rc=$?
    printf '%s\n' "${ecc}"
    echo
    if [ "${ecc_rc}" -ne 0 ]; then
        echo "[WARNING] uncorrected-ECC query unavailable on this driver (exit ${ecc_rc})."
        warn=1
    else
        local ecc_bad
        ecc_bad=$(printf '%s\n' "${ecc}" | awk -F',' 'NR>1 {
            for (i=1;i<=NF;i++) gsub(/^[ \t]+|[ \t]+$/,"",$i);
            if ($3~/^[0-9]+$/ && $3+0>0) print "  " $1 " (" $2 "): uncorr_ecc=" $3 }')
        if [ -n "${ecc_bad}" ]; then
            echo "[WARNING] GPU(s) report uncorrected ECC errors:"
            printf '%s\n' "${ecc_bad}"
            warn=1
        fi
    fi

    echo "--- remapped rows (uncorrectable / failure / pending) ---"
    local remap remap_rc
    remap=$(nvidia-smi --query-gpu=pci.bus_id,name,remapped_rows.uncorrectable,remapped_rows.failure,remapped_rows.pending --format=csv 2>&1)
    remap_rc=$?
    printf '%s\n' "${remap}"
    echo
    if [ "${remap_rc}" -ne 0 ]; then
        echo "[INFO] remapped-rows query unavailable on this nvidia-smi (exit ${remap_rc})."
        echo "       Row remapping is an Ampere+ feature, so this usually means an older driver,"
        echo "       not a hardware limit; a current driver would expose it. Skipping remap check."
    else
        # uncorrectable is a count; failure/pending are Yes/No booleans.
        local remap_bad
        remap_bad=$(printf '%s\n' "${remap}" | awk -F',' 'NR>1 {
            for (i=1;i<=NF;i++) gsub(/^[ \t]+|[ \t]+$/,"",$i);
            if (($3~/^[0-9]+$/ && $3+0>0) || toupper($4)=="YES" || toupper($5)=="YES")
                print "  " $1 " (" $2 "): remap_uncorr=" $3 " remap_fail=" $4 " remap_pending=" $5 }')
        if [ -n "${remap_bad}" ]; then
            echo "[WARNING] GPU(s) report uncorrectable row remaps, remap failures, or pending remaps:"
            printf '%s\n' "${remap_bad}"
            warn=1
        fi
    fi

    echo
    if [ "${warn}" -eq 0 ]; then
        echo "[OK] No uncorrected ECC errors or row-remap failures/pending reported."
    fi
}

run_step "8c" "nvidia-smi ECC + remapped rows - GPU memory health" \
    "$(declare -f check_gpu_ecc_remap); check_gpu_ecc_remap"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_nvlink_state() {
    command -v nvidia-smi > /dev/null 2>&1 || {
        echo "nvidia-smi not found"
        return 127
    }

    # Capture nvlink -s once and reuse it for display and both counts: a link
    # flapping between separate calls would otherwise desync display vs analysis,
    # and it avoids re-running on large topologies.
    local nvlink nvlink_rc
    nvlink=$(nvidia-smi nvlink -s 2>&1)
    nvlink_rc=$?
    echo "--- nvidia-smi nvlink -s ---"
    printf '%s\n' "${nvlink}"
    echo

    if [ "${nvlink_rc}" -ne 0 ]; then
        echo "[INFO] nvidia-smi nvlink -s failed (exit ${nvlink_rc}); skipping NVLink checks (not treated as healthy)."
        return 0
    fi

    local inactive up
    inactive=$(printf '%s\n' "${nvlink}" | grep -ciE 'inactive' || true)
    if [ "${inactive}" -gt 0 ]; then
        echo "[WARNING] ${inactive} NVLink(s) reported inactive."
    fi

    # On NVL72 platforms (GB200/GB300, 4 GPUs x 18 links) a healthy node reports 72 up links.
    case "${GPU_TYPE:-}" in
        gb200 | gb300)
            up=$(printf '%s\n' "${nvlink}" | grep -c 'GB/s' || true)
            echo "NVLink links reporting bandwidth: ${up} (expected 72 on ${GPU_TYPE} NVL72)."
            if [ "${up}" -eq 72 ]; then
                echo "[OK] 72 NVLinks up on ${GPU_TYPE}."
            else
                echo "[WARNING] Expected 72 NVLinks up on ${GPU_TYPE}, found ${up}."
            fi

            # Fabric (NVSwitch) registration must complete for multi-node NVLink.
            # fabric.state/status are GB/NVSwitch fields, so only queried here.
            # Exit-code guarded like the ECC/remap queries: an older nvidia-smi
            # that rejects the field must not be read as a passing fabric state.
            local fabric fabric_rc fabric_bad
            fabric=$(nvidia-smi --query-gpu=pci.bus_id,name,fabric.state,fabric.status --format=csv 2>&1)
            fabric_rc=$?
            echo "--- per-GPU fabric state/status ---"
            printf '%s\n' "${fabric}"
            echo
            if [ "${fabric_rc}" -ne 0 ]; then
                echo "[INFO] fabric-state query unavailable on this nvidia-smi (exit ${fabric_rc}); skipping fabric check."
            else
                fabric_bad=$(printf '%s\n' "${fabric}" | awk -F',' 'NR>1 {
                    gsub(/^[ \t]+|[ \t]+$/,"",$3); gsub(/^[ \t]+|[ \t]+$/,"",$4);
                    if ($3!="Completed" || $4!="Success")
                        print "  " $1 ": fabric.state=" $3 " fabric.status=" $4
                }')
                if [ -n "${fabric_bad}" ]; then
                    echo "[WARNING] GPU fabric not fully registered (expected state=Completed, status=Success):"
                    printf '%s\n' "${fabric_bad}"
                else
                    echo "[OK] GPU fabric registration complete on all GPUs (state=Completed, status=Success)."
                fi
            fi
            ;;
        *)
            if [ "${inactive}" -eq 0 ]; then
                echo "[OK] No NVLinks reported inactive."
            fi
            ;;
    esac
}

run_step "8d" "nvidia-smi nvlink -s + fabric state - NVLink/fabric up" \
    "$(declare -f check_nvlink_state); check_nvlink_state"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_nvlink_ber() {
    command -v nvidia-smi > /dev/null 2>&1 || {
        echo "nvidia-smi not found"
        return 127
    }

    # Capture with exit status so a failed nvlink -e isn't mistaken for "no BER
    # of concern" (empty grep). Each link reports "Effective BER:" and
    # "Symbol BER:"; 15e-255 is the healthy baseline (effectively zero).
    local raw raw_rc
    raw=$(nvidia-smi nvlink -e 2>&1)
    raw_rc=$?
    if [ "${raw_rc}" -ne 0 ]; then
        printf '%s\n' "${raw}"
        echo "[INFO] nvidia-smi nvlink -e failed (exit ${raw_rc}); skipping BER check (not treated as healthy)."
        return 0
    fi

    local out
    out=$(printf '%s\n' "${raw}" | grep 'BER:' | grep -v '15e-255' || true)
    if [ -z "${out}" ]; then
        echo "[OK] No NVLink BER values of concern reported."
    else
        printf '%s\n' "${out}"
        echo
        echo "[WARNING] Non-baseline NVLink BER values reported (see above)."
    fi
}

run_step "8e" "nvidia-smi nvlink -e - NVLink bit-error-rate" \
    "$(declare -f check_nvlink_ber); check_nvlink_ber"
run_step "9" "sysctl kernel.numa_balancing - automatic NUMA balancing" \
    "val=\$(sysctl -n kernel.numa_balancing) && printf 'kernel.numa_balancing = %s (%s)\n' \"\${val}\" \"\$([ \"\${val}\" = 0 ] && echo disabled || echo enabled)\""

run_step "10" "ibv_devinfo - InfiniBand HCA device info" "ibv_devinfo"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_ib_port_state() {
    local base=/sys/class/infiniband
    [ -d "${base}" ] || {
        echo "No InfiniBand devices found (${base} absent)"
        return
    }

    local dev port state phys any_down=0
    for dev in "${base}"/*; do
        [ -e "${dev}" ] || continue
        for port in "${dev}"/ports/*; do
            [ -d "${port}" ] || continue
            state=$(cat "${port}/state" 2> /dev/null)
            phys=$(cat "${port}/phys_state" 2> /dev/null)
            printf '  %s port %s: state=%s phys_state=%s\n' \
                "$(basename "${dev}")" "$(basename "${port}")" "${state}" "${phys}"
            case "${state}" in *ACTIVE*) ;; *) any_down=1 ;; esac
        done
    done
    echo
    if [ "${any_down}" -eq 0 ]; then
        echo "[OK] All InfiniBand ports report state ACTIVE."
    else
        echo "[WARNING] One or more InfiniBand ports are not ACTIVE (see above)."
        echo "          This may be normal -- not every port is necessarily wired or in use -- or it"
        echo "          may indicate a link problem. Confirm the set of ports that should be active"
        echo "          for this system with your cluster administrator."
    fi
}

run_step "10b" "IB port state - /sys/class/infiniband port up/active" \
    "$(declare -f check_ib_port_state); check_ib_port_state"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_ibgda_readiness() {
    # IBGDA (InfiniBand GPUDirect Async -- NVSHMEM's GPU-initiated transport) can run two ways:
    #   1. Default path  -> needs the nvidia kernel module loaded with
    #      PeerMappingOverride=1 (NVreg_RegistryDwords="PeerMappingOverride=1;").
    #   2. CPU-assisted path (NVSHMEM 3.0+, NVSHMEM_IBGDA_NIC_HANDLER=cpu) -> does NOT need
    #      PeerMappingOverride, but relies on the GDRCopy kernel driver (gdrdrv).
    # Either path is sufficient, so only warn when neither is satisfied.

    local peer_ok=0 gdr_ok=0 regdwords=""

    # --- Method 1: PeerMappingOverride on the nvidia kernel module ---
    # Anchor on the exact param: a bare 'RegistryDwords' substring also matches the
    # distinct 'RegistryDwordsPerDevice' line, which is just noise here.
    if [ -r /proc/driver/nvidia/params ]; then
        regdwords=$(grep -iE '^[[:space:]]*RegistryDwords[[:space:]]*:' /proc/driver/nvidia/params 2> /dev/null)
    fi
    if [ -r /sys/module/nvidia/parameters/NVreg_RegistryDwords ]; then
        regdwords="${regdwords} $(cat /sys/module/nvidia/parameters/NVreg_RegistryDwords 2> /dev/null)"
    fi
    printf '  nvidia RegistryDwords: %s\n' "${regdwords:-(unavailable)}"
    if printf '%s' "${regdwords}" | grep -Eqi 'PeerMappingOverride[[:space:]]*=[[:space:]]*1'; then
        peer_ok=1
    fi

    # --- Method 2: GDRCopy kernel driver (gdrdrv) for the CPU-assisted path ---
    if [ -e /dev/gdrdrv ]; then
        gdr_ok=1
        echo "  GDRCopy: /dev/gdrdrv present"
    elif lsmod 2> /dev/null | grep -q '^gdrdrv'; then
        gdr_ok=1
        echo "  GDRCopy: gdrdrv kernel module loaded"
    else
        echo "  GDRCopy: gdrdrv not found (no /dev/gdrdrv, module not loaded)"
    fi
    echo

    if [ "${peer_ok}" -eq 1 ]; then
        echo "[OK] PeerMappingOverride=1 set -> default IBGDA path is available."
    elif [ "${gdr_ok}" -eq 1 ]; then
        echo "[OK] PeerMappingOverride not set, but GDRCopy (gdrdrv) is present ->"
        echo "     CPU-assisted IBGDA path is available (NVSHMEM_IBGDA_NIC_HANDLER=cpu)."
    else
        echo "[WARNING] No viable IBGDA path detected."
        echo "          Default IBGDA needs the nvidia module loaded with"
        echo '          NVreg_RegistryDwords="PeerMappingOverride=1;"; the CPU-assisted path'
        echo "          needs the GDRCopy kernel driver (gdrdrv). Neither was found, so"
        echo "          NVSHMEM GPU-initiated workloads may fall back to slower transports."
    fi
}

run_step "10c" "IBGDA readiness - PeerMappingOverride or GDRCopy for NVSHMEM GPU-initiated transport" \
    "$(declare -f check_ibgda_readiness); check_ibgda_readiness"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_slurm_topology_config() {
    if ! command -v scontrol > /dev/null 2>&1; then
        echo "scontrol command not found"
        return 127
    fi

    local config
    config=$(scontrol show config)
    local rc=$?
    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] scontrol show config returned exit code ${rc}."
        return "${rc}"
    fi

    echo "--- Slurm topology configuration ---"
    printf "%s\n" "${config}" | grep -Ei "^[[:space:]]*(TopologyPlugin|TopologyParam)[[:space:]]*=" \
        || echo "(no TopologyPlugin or TopologyParam entries found)"
}

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_slurm_topology() {
    if ! command -v scontrol > /dev/null 2>&1; then
        echo "scontrol command not found"
        return 127
    fi

    local topology
    topology=$(scontrol show topology)
    local rc=$?
    if [[ -n ${topology} ]]; then
        printf "%s\n" "${topology}"
    fi

    echo
    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] scontrol show topology returned exit code ${rc}."
        return "${rc}"
    fi

    if [[ -z ${topology//[[:space:]]/} ]]; then
        echo "[FAILED] scontrol show topology returned empty output."
        echo "         Slurm topology should be configured for benchmark clusters that"
        echo "         span more than one rack or network switch."
        return 1
    fi

    echo "[OK] scontrol show topology returned topology data."
}

run_step "11a" "scontrol show config - Slurm topology plugin configuration" \
    "$(declare -f check_slurm_topology_config); check_slurm_topology_config"
run_step "11b" "scontrol show topology - Slurm topology configuration is present" \
    "$(declare -f check_slurm_topology); check_slurm_topology"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_slurm_mpi_pmix() {
    local output rc
    output=$(srun --mpi=list 2>&1)
    rc=$?
    printf '%s\n' "${output}"
    echo

    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] srun --mpi=list returned exit code ${rc}."
        return "${rc}"
    fi

    if printf '%s\n' "${output}" | grep -qiE "(^|[[:space:]])pmix"; then
        echo "[OK] PMIx MPI plugin is available."
    else
        echo "[FAILED] PMIx MPI plugin not listed by 'srun --mpi=list'."
        echo "         LLMB recipes invoke Slurm with --mpi=pmix; Slurm must be built with --with-pmix."
        return 1
    fi
}

run_step "11c" "srun --mpi=list - Slurm PMIx MPI plugin available" \
    "$(declare -f check_slurm_mpi_pmix); check_slurm_mpi_pmix"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_enroot_conf() {
    local conf="/etc/enroot/enroot.conf"

    echo "--- ${conf} (non-comment lines) ---"
    if [ ! -f "${conf}" ]; then
        echo "${conf} not found"
        return
    fi

    grep -v "^[[:space:]]*#" "${conf}" | grep -v "^[[:space:]]*$" \
        || echo "(all lines commented or empty)"

    echo
    echo "--- enroot.conf diagnostics ---"
    if grep -Eqi "^[[:space:]]*ENROOT_ROOTFS_WRITABLE[[:space:]=]+(y|yes)[[:space:]]*$" "${conf}"; then
        echo "[OK] ENROOT_ROOTFS_WRITABLE enabled."
    else
        echo "[WARNING] ENROOT_ROOTFS_WRITABLE not enabled (expected 'y' or 'yes')."
        echo "          LLMB setup and launch paths expect writable container filesystems."
    fi
    if grep -Eqi "^[[:space:]]*ENROOT_REMAP_ROOT[[:space:]=]+(y|yes)[[:space:]]*$" "${conf}"; then
        echo "[OK] ENROOT_REMAP_ROOT enabled."
    else
        echo "[WARNING] ENROOT_REMAP_ROOT not enabled (expected 'y' or 'yes')."
        echo "          LLMB recipes recommend remapping root in containerized Slurm jobs."
    fi
}

# shellcheck disable=SC2317  # invoked indirectly via declare -f
enroot_environ_has_nonempty_assignment() {
    local envdir="$1"
    local var_name="$2"
    local f line value

    for f in "${envdir}"/*; do
        [ -f "$f" ] || continue
        while IFS= read -r line || [ -n "${line}" ]; do
            [[ ${line} =~ ^[[:space:]]*${var_name}[[:space:]]*= ]] || continue
            value="${line#*=}"
            value="${value#"${value%%[![:space:]]*}"}"
            value="${value%"${value##*[![:space:]]}"}"
            case "${value}" in
                \"*\" | \'*\')
                    value="${value:1:${#value}-2}"
                    value="${value#"${value%%[![:space:]]*}"}"
                    value="${value%"${value##*[![:space:]]}"}"
                    ;;
            esac
            [ -n "${value}" ] && return 0
        done < "$f"
    done

    return 1
}

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_enroot_environ() {
    local envdir="/etc/enroot/environ.d"

    echo "--- ${envdir}/ contents ---"
    if [ ! -d "${envdir}" ]; then
        echo "${envdir}/ directory not found"
        return
    fi

    local env_files=("${envdir}"/*.env)
    if [ ! -e "${env_files[0]}" ]; then
        echo "(no .env files found)"
        env_files=()
    fi

    for f in "${envdir}"/*; do
        [ -f "$f" ] && printf "# %s\n" "$(basename "$f")" && cat "$f" && echo
    done

    echo "--- environ.d diagnostics ---"
    if [ "${#env_files[@]}" -le 1 ]; then
        echo "[WARNING] Only ${#env_files[@]} env file(s) in environ.d -- likely bare defaults."
        echo "          Cluster-specific files (NCCL, Mellanox, NVIDIA, UCX) appear missing."
        echo "          This is a common cause of hard-to-debug failures when containers"
        echo "          do not inherit the required network/GPU environment."
    fi
    if enroot_environ_has_nonempty_assignment "${envdir}" "MELLANOX_VISIBLE_DEVICES"; then
        echo "[OK] MELLANOX_VISIBLE_DEVICES is configured in environ.d (restricts which HCAs are exposed to the container)."
    elif enroot_environ_has_nonempty_assignment "${envdir}" "NCCL_IB_HCA" \
        && enroot_environ_has_nonempty_assignment "${envdir}" "NVSHMEM_HCA_LIST"; then
        echo "[OK] NCCL_IB_HCA and NVSHMEM_HCA_LIST are configured in environ.d."
    elif enroot_environ_has_nonempty_assignment "${envdir}" "NCCL_IB_HCA"; then
        echo "[WARNING] NCCL_IB_HCA is configured, but NVSHMEM_HCA_LIST is not."
        echo "          NCCL settings do not constrain every recipe; NVSHMEM-based workloads"
        echo "          may still select other HCAs unless NVSHMEM_HCA_LIST is configured too."
    else
        echo "[WARNING] HCA selection is incomplete in environ.d."
        echo "          Set MELLANOX_VISIBLE_DEVICES, or set both NCCL_IB_HCA and NVSHMEM_HCA_LIST."
        echo "          This matters for NVSHMEM-based recipes, which do not rely on NCCL for HCA selection."
    fi
}

run_step "12a" "enroot config - enroot.conf" \
    "$(declare -f check_enroot_conf); check_enroot_conf"
run_step "12b" "enroot config - environ.d" \
    "$(declare -f enroot_environ_has_nonempty_assignment); $(declare -f check_enroot_environ); check_enroot_environ"

run_step "13a" "srun container nvidia-smi - GPU visibility inside container (${IMAGE})" \
    "srun --nodes=1 --ntasks=1 --container-image '${IMAGE}' --container-writable --no-container-mount-home nvidia-smi"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_container_slurm_pytorch_hook() {
    local image="$1"
    local output rc

    output=$(srun --nodes=1 --ntasks=1 \
        --container-image "${image}" \
        --container-writable --no-container-mount-home \
        bash -c 'env | grep -E "^(MASTER_ADDR|MASTER_PORT|WORLD_SIZE|LOCAL_RANK|RANK)=" | sort || true' 2>&1)
    rc=$?
    printf '%s\n' "${output}"
    echo

    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] srun bash inside container returned exit code ${rc}."
        return "${rc}"
    fi

    local missing=()
    local var
    for var in MASTER_ADDR MASTER_PORT WORLD_SIZE; do
        if ! printf '%s\n' "${output}" | grep -q "^${var}=[^[:space:]]"; then
            missing+=("${var}")
        fi
    done

    if [[ ${#missing[@]} -eq 0 ]]; then
        echo "[OK] MASTER_ADDR, MASTER_PORT, and WORLD_SIZE are set inside the container."
        echo "     Enroot extra hook '50-slurm-pytorch.sh' appears to be installed and configured."
    else
        echo "[FAILED] Missing PyTorch hook vars inside the container: ${missing[*]}"
        echo "         Enroot extra hook '50-slurm-pytorch.sh' is missing or not configured."
        echo "         LLMB recipes rely on this hook to populate MASTER_ADDR/MASTER_PORT/WORLD_SIZE"
        echo "         from Slurm env vars at container start."
        echo "         See https://github.com/NVIDIA/enroot/tree/main/conf/hooks/extra"
        return 1
    fi
}

run_step "13b" "srun container env - 50-slurm-pytorch.sh hook sets MASTER_ADDR, MASTER_PORT, WORLD_SIZE (${IMAGE})" \
    "$(declare -f check_container_slurm_pytorch_hook); check_container_slurm_pytorch_hook '${IMAGE}'"

# shellcheck disable=SC2317  # invoked indirectly via declare -f
check_container_imex() {
    local image="$1"
    local output rc

    # IMEX is a GB-series (NVL72) feature. Gate on GPU_TYPE (provided by
    # llmb-run), not CPU architecture, and skip before launching the container
    # on non-GB platforms.
    case "${GPU_TYPE:-}" in
        gb200 | gb300) ;;
        *)
            echo "GPU type '${GPU_TYPE:-unset}' is not GB-series; IMEX is not used."
            echo "[OK] Non-GB platform; IMEX check not applicable."
            return 0
            ;;
    esac

    # Check what the job's container actually sees, independent of how IMEX is
    # set up on the host (systemd service, Slurm prolog, etc.). Cross-node
    # NVLink (NVLink IPC) needs IMEX channel devices exposed inside the container.
    # shellcheck disable=SC2016  # single quotes intentional: body runs inside the container, not expanded by the host shell
    output=$(srun --nodes=1 --ntasks=1 \
        --container-image "${image}" \
        --container-writable --no-container-mount-home \
        bash -c 'echo "channels:"; ls -1 /dev/nvidia-caps-imex-channels/ 2>/dev/null || echo "(none)"' 2>&1)
    rc=$?
    printf '%s\n' "${output}"
    echo

    if [[ ${rc} -ne 0 ]]; then
        echo "[FAILED] srun IMEX check inside container returned exit code ${rc}."
        return "${rc}"
    fi

    if printf '%s\n' "${output}" | grep -qE '^channel[0-9]'; then
        echo "[OK] IMEX channel devices are visible inside the container."
    else
        echo "[WARNING] No IMEX channel devices visible inside the container."
        echo "          GB-series cross-node NVLink (NVLink IPC) requires IMEX channels exposed"
        echo "          to the container; check the IMEX setup and container device passthrough."
    fi
}

run_step "13c" "srun container IMEX - IMEX channels visible inside the container (${IMAGE})" \
    "$(declare -f check_container_imex); check_container_imex '${IMAGE}'"

print_banner "System Info Collection Summary"
echo "Failed non-fatal steps: ${FAILED_STEPS}"
echo "Completed at: $(date -Iseconds)"

exit 0
