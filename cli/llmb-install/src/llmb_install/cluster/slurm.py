# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""SLURM cluster configuration utilities for LLMB installer."""

import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple


def parse_gpu_gres(gres_output: str) -> Optional[int]:
    """Extract the GPU count from a SLURM GRES string.

    Accepted examples:
        gpu:8
        gpu:a100:8
        gpu:8(S:0-1)
        gpu:a100:8(S:0-1)
        gpu:8,mib:100
        gpu:a100_3g.20gb:2

    Returns an integer within the range 1-8 or *None* when parsing fails.
    """
    if not gres_output or gres_output == "(null)" or "gpu:" not in gres_output:
        return None

    # Keep the substring after "gpu:" then strip any extras after ',' or '('
    gpu_part = gres_output.split("gpu:", 1)[1]
    for sep in (",", "("):
        gpu_part = gpu_part.split(sep, 1)[0]

    # In cases like 'gpu:a100:8' keep only the numeric part after the last ':'
    gpu_part = gpu_part.split(":")[-1].strip()

    if not gpu_part.isdigit():
        return None

    count = int(gpu_part)
    return count if 1 <= count <= 8 else None


def augment_env_for_job_type(
    env: Dict[str, str], job_type: str, slurm_info: Dict[str, Any], requires_gpus: bool = False
):
    """Inject cluster-specific SBATCH/SLURM variables.

    Task metadata must *not* override these values – they are entirely
    cluster/user-specific.  Therefore we **overwrite** any existing keys.

    Args:
        env: Environment dictionary to modify
        job_type: Type of job (local, nemo2, sbatch, srun)
        slurm_info: SLURM configuration dictionary
        requires_gpus: Whether this task requires GPU resources
    """
    account = slurm_info["slurm"].get("account", "")

    # Select partition and GRES based on GPU requirements
    if requires_gpus:
        partition = slurm_info["slurm"].get("gpu_partition")
        gres = slurm_info["slurm"].get("gpu_partition_gres")
    else:
        partition = slurm_info["slurm"].get("cpu_partition")
        gres = slurm_info["slurm"].get("cpu_partition_gres")

    if job_type in ("nemo2", "sbatch", "srun"):
        if not account:
            raise ValueError(
                f"SLURM account must be set for job_type '{job_type}'. Please provide that information during installation."
            )
        if not partition:
            partition_type = "GPU" if requires_gpus else "CPU"
            raise ValueError(
                f"SLURM {partition_type} partition must be set for job_type '{job_type}'. Please provide that information during installation."
            )

    if job_type in ("nemo2", "sbatch"):
        env["SBATCH_ACCOUNT"] = account
        env["SBATCH_PARTITION"] = partition
        if gres is not None:
            env["SBATCH_GPUS_PER_NODE"] = str(gres)
    elif job_type == "srun":
        env["SLURM_ACCOUNT"] = account
        env["SLURM_PARTITION"] = partition
        if gres is not None:
            env["SLURM_GPUS_PER_NODE"] = str(gres)


def get_cluster_name() -> Optional[str]:
    """Get the SLURM cluster name if available.

    Returns:
        Optional[str]: The cluster name if found, None otherwise
    """
    try:
        result = subprocess.run(["scontrol", "show", "config"], capture_output=True, text=True, check=True)

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.lower().startswith('clustername'):
                # Parse "ClusterName = cluster" format
                if '=' in line:
                    cluster_name = line.split('=', 1)[1].strip()
                    return cluster_name if cluster_name else None

        return None

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_user_accounts() -> List[str]:
    """Get SLURM accounts associated with the current user.

    Returns:
        List[str]: List of account names, empty if none found or error occurred
    """
    try:
        username = os.environ.get('USER') or os.environ.get('USERNAME')
        if not username:
            result = subprocess.run(["whoami"], capture_output=True, text=True, check=True)
            username = result.stdout.strip()

        # Get accounts associated with the current user
        result = subprocess.run(
            ["sacctmgr", "show", "assoc", f"user={username}", "format=account", "-p", "-n"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse accounts and remove the pipe character at the end
        accounts = [a.strip().rstrip('|') for a in result.stdout.strip().split('\n') if a.strip()]

        return sorted(set(accounts))

    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_available_partitions() -> Tuple[List[str], Optional[str]]:
    """Get available SLURM partitions.

    Returns:
        Tuple[List[str], Optional[str]]: (partitions, default_partition)
    """
    try:
        result = subprocess.run(["sinfo", "--noheader", "-o", "%P"], capture_output=True, text=True, check=True)

        # Parse the output to identify the default partition (marked with *)
        raw_partitions = result.stdout.strip().split('\n')
        partitions = []
        default_partition = None

        for p in raw_partitions:
            p = p.strip()
            if p.endswith('*'):
                default_partition = p.rstrip('*')
                partitions.append(default_partition)
            else:
                partitions.append(p)

        return sorted(set(partitions)), default_partition

    except (subprocess.SubprocessError, FileNotFoundError):
        return [], None


def validate_account(account_input: str, available_accounts: List[str]) -> str:
    """Validate account input.

    Args:
        account_input: User's account input
        available_accounts: List of available accounts

    Returns:
        str: 'valid' if valid, 'show_all' if should show all accounts,
             or 'invalid:<message>' if invalid
    """
    if not account_input:
        return 'valid'

    if account_input == '?' and available_accounts and len(available_accounts) > 10:
        return 'show_all'

    if available_accounts and account_input not in available_accounts:
        return f"invalid:'{account_input}' is not in the list of available accounts."

    return 'valid'


def validate_partition(partition_input: str, available_partitions: List[str]) -> str:
    """Validate partition input (supports comma-separated lists).

    Args:
        partition_input: User's partition input (single or comma-separated list)
        available_partitions: List of available partitions

    Returns:
        str: 'valid' if valid, 'show_all' if should show all partitions,
             or 'invalid:<message>' if invalid
    """
    if not partition_input:
        return 'valid'

    if partition_input == '?' and available_partitions and len(available_partitions) > 10:
        return 'show_all'

    # Support comma-separated partition lists (valid SLURM syntax)
    partitions_list = normalize_partition_list(partition_input)

    if not partitions_list:
        return 'valid'

    # If no available partitions list, accept anything (can't validate)
    if not available_partitions:
        return 'valid'

    # Check each partition in the list
    invalid_partitions = [p for p in partitions_list if p not in available_partitions]

    if invalid_partitions:
        if len(invalid_partitions) == 1:
            return f"invalid:'{invalid_partitions[0]}' is not in the list of available partitions."
        else:
            invalid_str = "', '".join(invalid_partitions)
            return f"invalid:Partitions not found: '{invalid_str}'."

    return 'valid'


def get_default_cpu_partition(partitions: List[str]) -> Optional[str]:
    """Get a default CPU partition from available partitions.

    Args:
        partitions: List of available partitions

    Returns:
        Optional[str]: Default CPU partition name, if found
    """
    cpu_partitions = [p for p in partitions if p.startswith('cpu')]
    return cpu_partitions[0] if cpu_partitions else None


def normalize_partition_list(partition_input: str) -> list[str]:
    """Normalize comma-separated partition text into a deduplicated list of names."""
    if not partition_input:
        return []
    return list(dict.fromkeys(partition.strip() for partition in partition_input.split(',') if partition.strip()))


def _sample_node_names(node_names: list[str], sample_size: int = 5) -> list[str]:
    """Return up to sample_size unique node names, preserving sinfo order."""
    if sample_size <= 0:
        return []
    sampled = []
    seen = set()
    for node_name in node_names:
        if node_name in seen:
            continue
        sampled.append(node_name)
        seen.add(node_name)
        if len(sampled) >= sample_size:
            break
    return sampled


def _parse_node_architectures(scontrol_output: str) -> dict[str, str]:
    """Parse node architectures from ``scontrol show node -o`` output."""
    node_architectures = {}
    for line in scontrol_output.splitlines():
        node_match = re.search(r'(?:^|\s)NodeName=(\S+)', line)
        arch_match = re.search(r'(?:^|\s)Arch=(\S+)', line)
        if node_match and arch_match:
            node_architectures[node_match.group(1)] = arch_match.group(1)
    return node_architectures


def detect_partition_architecture(partition: str, sample_size: int = 5) -> Dict[str, Any]:
    """Detect the CPU architecture for a GPU partition using a bounded Slurm sample.

    Returns:
        Dict with detection details:
        {
            'architecture': Optional[str],  # x86_64 or aarch64 when confidently detected
            'reason': str,                  # detected, no_nodes, no_arch, mixed, unsupported, or error
            'nodes': list[str],             # sampled nodes queried with scontrol
            'architectures': dict[str,str], # parsed node -> arch values
            'error': Optional[str],         # command error detail when available
        }
    """
    empty_result = {
        "architecture": None,
        "reason": "error",
        "nodes": [],
        "architectures": {},
        "error": None,
    }

    partitions = normalize_partition_list(partition)
    if not partitions:
        return {**empty_result, "reason": "no_nodes"}

    try:
        sinfo_result = subprocess.run(
            ["sinfo", "-N", "--noheader", "-p", ",".join(partitions), "-o", "%N"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        return {**empty_result, "error": str(e)}

    nodes = _sample_node_names([line.strip() for line in sinfo_result.stdout.splitlines() if line.strip()], sample_size)
    if not nodes:
        return {**empty_result, "reason": "no_nodes"}

    try:
        scontrol_result = subprocess.run(
            ["scontrol", "-o", "show", "node", ",".join(nodes)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        return {**empty_result, "nodes": nodes, "error": str(e)}

    architectures = _parse_node_architectures(scontrol_result.stdout)
    if not architectures:
        return {**empty_result, "reason": "no_arch", "nodes": nodes}

    unique_architectures = set(architectures.values())
    if len(unique_architectures) > 1:
        return {
            **empty_result,
            "reason": "mixed",
            "nodes": nodes,
            "architectures": architectures,
        }

    architecture = next(iter(unique_architectures))
    if architecture not in {"x86_64", "aarch64"}:
        return {
            **empty_result,
            "reason": "unsupported",
            "nodes": nodes,
            "architectures": architectures,
        }

    return {
        "architecture": architecture,
        "reason": "detected",
        "nodes": nodes,
        "architectures": architectures,
        "error": None,
    }


def detect_partition_gres(partitions: list[str]) -> Dict[str, Dict[str, Any]]:
    """Detect GRES information for the given partitions using sinfo.

    Each entry in *partitions* may be a single name (``"gpu1"``) or a
    comma-separated composite (``"gpu1,gpu2"``).  Composites are expanded
    for the sinfo query and the results are consolidated back under the
    original composite key.  All partitions within a composite must share
    the same GPU count.

    Returns:
        Dict mapping the original partition strings to their GRES info:
        {
            'partition_name': {
                'gpu_count': Optional[int],       # GPU count backed by the most nodes
                'has_gpu_lines': bool,            # at least one line contained 'gpu:'
                'unparseable_lines': list[str],   # gpu lines we failed to parse
                'is_heterogeneous': bool,           # multiple distinct counts detected
                'nodes_by_gpu_count': dict[int,int] # {gpu_count: node_count}, desc by gpu
            }
        }

    Raises:
        SystemExit: If sinfo is not available (hard error)
        RuntimeError: If sinfo command fails for other reasons
        ValueError: If partitions within a composite have inconsistent GPU counts
    """
    partition_to_requested_names: dict[str, list[str]] = {
        partition: normalize_partition_list(partition) for partition in partitions
    }

    # Keep a stable order for sinfo query while removing duplicates.
    expanded_partitions = list(dict.fromkeys(name for names in partition_to_requested_names.values() for name in names))

    partition_info = {
        partition: {
            "gpu_entries": [],  # list of (gpu_count, node_count) tuples
            "has_gpu_lines": False,
            "unparseable_lines": [],
        }
        for partition in expanded_partitions
    }

    try:
        result = subprocess.run(
            ["sinfo", "--noheader", "-p", ",".join(expanded_partitions), "-o", "%P,%G,%D"],
            capture_output=True,
            text=True,
            check=True,
        )
        sinfo_lines = result.stdout.strip().splitlines()
    except FileNotFoundError:
        print("\nError: 'sinfo' command not found.")
        print("This installer must be run on a system with SLURM installed.")
        print("Please run this installer on a SLURM login node.")
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as e:
        print(f"\nWarning: 'sinfo' command failed with return code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr.strip()}")
        print("Cannot auto-detect GRES information. Will prompt for manual input.")
        raise RuntimeError(f"sinfo command failed: {e}") from e

    # Parse sinfo output (format: partition,gres,node_count)
    for line in sinfo_lines:
        if not line.strip() or "," not in line:
            continue

        # Split node count (last field) first, then partition from gres
        rest, _, node_count_raw = line.rstrip().rpartition(",")
        if not rest:
            continue
        partition_name, _, gres_raw = rest.partition(",")
        if not gres_raw:
            continue
        partition_name = partition_name.strip().rstrip("*")
        gres_raw = gres_raw.strip()

        if partition_name not in partition_info:
            continue

        try:
            node_count = int(node_count_raw)
        except ValueError:
            continue

        if "gpu:" in gres_raw:
            partition_info[partition_name]["has_gpu_lines"] = True

        gpu_count = parse_gpu_gres(gres_raw)
        if gpu_count is not None:
            partition_info[partition_name]["gpu_entries"].append((gpu_count, node_count))
        elif "gpu:" in gres_raw:
            # Keep track of lines we failed to parse that should have GPU info
            partition_info[partition_name]["unparseable_lines"].append(gres_raw)

    # Consolidate per-sinfo results back into the original (possibly composite) keys.
    composite_partition_info = {}
    for composite_partition, individual_partitions in partition_to_requested_names.items():
        if not individual_partitions:
            composite_partition_info[composite_partition] = {
                "gpu_count": None,
                "has_gpu_lines": False,
                "unparseable_lines": [],
                "is_heterogeneous": False,
                "nodes_by_gpu_count": {},
            }
            continue

        merged = {"gpu_entries": [], "has_gpu_lines": False, "unparseable_lines": []}
        per_partition_gpu_count = {}

        for name in individual_partitions:
            info = partition_info[name]
            merged["gpu_entries"].extend(info["gpu_entries"])
            merged["has_gpu_lines"] = merged["has_gpu_lines"] or info["has_gpu_lines"]
            merged["unparseable_lines"].extend(info["unparseable_lines"])
            # Aggregate node counts per gpu_count, then pick the dominant one
            nodes_by_gpu: dict[int, int] = {}
            for gc, nc in info["gpu_entries"]:
                nodes_by_gpu[gc] = nodes_by_gpu.get(gc, 0) + nc
            per_partition_gpu_count[name] = (
                max(nodes_by_gpu, key=lambda g: (nodes_by_gpu[g], g)) if nodes_by_gpu else None
            )

        if len(set(per_partition_gpu_count.values())) > 1:
            details = ", ".join(f"{p}={c if c is not None else 'None'}" for p, c in per_partition_gpu_count.items())
            raise ValueError(
                f"Inconsistent GPU GRES across partition list '{composite_partition}': {details}. "
                "Use partitions with matching GRES settings."
            )

        # Aggregate node counts per gpu_count
        nodes_by_gpu: dict[int, int] = {}
        for gc, nc in merged["gpu_entries"]:
            nodes_by_gpu[gc] = nodes_by_gpu.get(gc, 0) + nc

        if not nodes_by_gpu:
            gpu_count = None
            is_heterogeneous = False
        else:
            gpu_count = max(nodes_by_gpu, key=lambda gc: (nodes_by_gpu[gc], gc))
            is_heterogeneous = len(nodes_by_gpu) > 1

        composite_partition_info[composite_partition] = {
            "gpu_count": gpu_count,
            "has_gpu_lines": merged["has_gpu_lines"],
            "unparseable_lines": merged["unparseable_lines"],
            "is_heterogeneous": is_heterogeneous,
            "nodes_by_gpu_count": dict(sorted(nodes_by_gpu.items(), reverse=True)),
        }

    return composite_partition_info
