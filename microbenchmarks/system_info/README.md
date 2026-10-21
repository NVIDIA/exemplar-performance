# Overview

Use this recipe to inspect the compute nodes before running training or inference benchmarks. It collects hardware and software information and flags configuration or health issues that could affect the run.

There are two groups of checks:

- **Universal host checks** inspect the machine itself: its CPUs, memory layout, GPUs and network connections.
- **Slurm checks** inspect the job environment: whether Slurm can start workload containers with the GPU and communication settings they need.

## How it runs

- **Slurm:** run the single command in [Run](#run). The script runs the host checks first, then the Slurm checks, and writes the results to a log.
- **Kubernetes:** run the host checks manually on each compute host. The Slurm checks do not apply.

Review the results and address reported issues before starting the full benchmark runs.

# Commands Collected

## Universal host checks

These checks apply to both Slurm and Kubernetes. For Kubernetes, run the applicable checks manually on each compute host, with administrator help where needed.

The host checks help you understand three areas:

- **CPU and memory:** which processors are installed, how memory is arranged, and which power and operating-system settings may affect performance.
- **GPU health and connections:** which GPUs are present, how they are connected, and whether they report memory errors or link problems.
- **Network setup:** which network devices and ports are present, whether their links are active, and whether driver settings allow GPU-initiated network communication.

<details>
<summary>View commands and warning criteria</summary>

### CPU and host setup

| Check                      | Command or source                                           | What it tells you                                             |
| -------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
| CPU hardware               | `lscpu`                                                     | Which CPUs are installed and how their cores are arranged.    |
| CPU power settings         | `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` | Which CPU frequency policy each core uses.                    |
| CPU idle states            | `/sys/devices/system/cpu/cpu0/cpuidle`                      | Which idle states CPU 0 exposes and whether they are enabled. |
| Attached hardware          | `lspci -v`                                                  | Which PCI devices are installed, including network adapters.  |
| CPU and memory layout      | `numactl -H`                                                | How CPUs and memory are grouped into NUMA nodes.              |
| Kernel settings            | `cat /proc/cmdline`                                         | Which settings were passed to Linux at startup.               |
| Virtualization             | `systemd-detect-virt`                                       | Whether the host runs in a virtualized environment.           |
| Memory page size           | `getconf PAGE_SIZE`                                         | The operating system's memory page size.                      |
| SMMU messages              | `dmesg \| grep -i smmu`                                     | Kernel messages about the system's I/O memory management.     |
| Automatic memory placement | `sysctl -n kernel.numa_balancing`                           | Whether Linux automatically moves memory between NUMA nodes.  |

**Warnings:** the collector warns if any core is not using the `performance` frequency governor, which can add frequency-scaling noise to benchmark results. It also warns if CPU 0 exposes two or fewer idle states or any are disabled. These settings can affect CPU boost clocks.

### GPU health and connections

| Check                    | Command or source                                                      | What it tells you                                                 |
| ------------------------ | ---------------------------------------------------------------------- | ----------------------------------------------------------------- |
| GPU information          | `nvidia-smi -q`                                                        | GPU hardware, firmware, driver and status information.            |
| GPU connection layout    | `nvidia-smi topo -m`                                                   | How GPUs and network adapters connect to each other and the CPUs. |
| GPU memory health        | `nvidia-smi --query-gpu=...ecc.errors.uncorrected...,remapped_rows...` | Reported memory errors and row-remapping status.                  |
| NVLink and fabric status | `nvidia-smi nvlink -s` + `--query-gpu=fabric.state,fabric.status`      | Whether GPU links and fabric connections are ready.               |
| NVLink error rate        | `nvidia-smi nvlink -e`                                                 | Reported bit-error rates on GPU links.                            |

**Warnings:** the collector flags uncorrected ECC errors, row-remap failures or pending remaps, inactive NVLinks, and non-baseline bit-error rates. On `gb200`/`gb300` NVL72 nodes, it also expects 72 active links. That link-count check uses `GPU_TYPE` and does not apply to other platforms.

### Network readiness

| Check                  | Command or source                                                                  | What it tells you                                                         |
| ---------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| InfiniBand adapters    | `ibv_devinfo`                                                                      | Which adapters are present and what they support.                         |
| Network port status    | `/sys/class/infiniband/*/ports/*/state`                                            | Whether each InfiniBand port is `ACTIVE`.                                 |
| NVIDIA driver settings | `/proc/driver/nvidia/params`, `/sys/module/nvidia/parameters/NVreg_RegistryDwords` | Whether `PeerMappingOverride=1` enables the default IBGDA path.           |
| GDRCopy availability   | `/dev/gdrdrv` or `lsmod`                                                           | Whether the `gdrdrv` driver is available for the CPU-assisted IBGDA path. |

**Warnings:** inactive ports are flagged, but unused ports may legitimately be down, such as HCAs wired to NVSwitch on B200/B300.

**IBGDA readiness:** this checks the prerequisites for NVSHMEM's GPU-initiated network communication. Either `PeerMappingOverride=1` or GDRCopy is sufficient for the collector's check; it warns if neither is present.

</details>

## Slurm checks

These checks help identify problems that could prevent a Slurm workload from starting or using the expected resources:

- **Job setup:** checks cluster topology and support for launching distributed jobs.
- **Container setup:** checks the settings used to start workload containers and expose network devices.
- **Access inside a container:** runs a test container to check GPU visibility and the settings and devices needed for communication between GPUs and nodes.

<details>
<summary>View Slurm commands and warning criteria</summary>

### Scheduler and job setup

| Check                   | Command                  | What it tells you                                                                   |
| ----------------------- | ------------------------ | ----------------------------------------------------------------------------------- |
| Scheduler settings      | `scontrol show config`   | Shows `TopologyPlugin` and `TopologyParam` for context.                             |
| Cluster topology        | `scontrol show topology` | Whether Slurm reports the cluster's topology; empty output is flagged as a failure. |
| Distributed-job support | `srun --mpi=list`        | Whether the required `pmix` plugin is available.                                    |

LLMB jobs use `--mpi=pmix`, so Slurm must be built with `--with-pmix`.

### Container configuration

| Check                      | Configuration source | What it tells you                                                                                 |
| -------------------------- | -------------------- | ------------------------------------------------------------------------------------------------- |
| Container startup settings | `enroot.conf`        | Whether the recommended `ENROOT_ROOTFS_WRITABLE` and `ENROOT_REMAP_ROOT` settings are configured. |
| Network-device settings    | `environ.d/`         | Which network devices are exposed to a workload container.                                        |

The network-device check flags bare defaults. It passes when `MELLANOX_VISIBLE_DEVICES` is configured, or both `NCCL_IB_HCA` and `NVSHMEM_HCA_LIST` are configured. `MELLANOX_VISIBLE_DEVICES` is preferred because it limits which adapters the container can see.

### Checks inside a container

These commands run through `srun` inside a container to check what the job can actually access.

| Check                       | Command                              | What it tells you                                      |
| --------------------------- | ------------------------------------ | ------------------------------------------------------ |
| GPU visibility              | `nvidia-smi`                         | Whether GPUs are visible inside the container.         |
| Distributed-job environment | `env \| grep MASTER_ADDR`            | Whether Enroot's PyTorch startup hook is active.       |
| Cross-node NVLink access    | `ls /dev/nvidia-caps-imex-channels/` | Whether the container can access IMEX channel devices. |

The [`50-slurm-pytorch.sh`](https://github.com/NVIDIA/enroot/tree/main/conf/hooks/extra) hook provides `MASTER_ADDR`, `MASTER_PORT` and `WORLD_SIZE`, which are required for distributed startup.

The IMEX check applies when `GPU_TYPE` is GB200/GB300. Missing channel devices produce a warning because cross-node NVLink communication requires them. The check uses the container's view, regardless of how IMEX is configured on the host, and is skipped on other platforms.

</details>

The Slurm collector treats command failures as non-fatal; failures are reported in output and execution continues.

# Run

Slurm only. The collector runs both check groups above.

```bash
cd $LLMB_INSTALL
llmb-run submit -w microbenchmark_system_info --scale <num_gpus_per_node>
```

# Output

The script writes output to standard SLURM output (`slurm-*.out`) under the experiment
directory created by `configured_sbatch`.

Review the warnings and failed checks in that log. The script continues collecting information when a check fails, so a completed job still needs its results reviewed.
