# Metrics

TaskTrace-capture records benchmark outcomes, timing, host activity, optional inference-server telemetry, and system information. Each completed sweep writes three primary result files:

| File                   | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `report.txt`           | Human-readable summary for a quick review.                 |
| `report.json`          | Structured per-concurrency summary and system information. |
| `scaling_results.json` | Detailed sweep results, including raw host samples.        |

The sections below describe the main fields, how they are calculated, and the scope of each measurement.

## `report.json` fields

Metrics under `concurrency.<level>` are reported once for each requested concurrency level.

| Category     | Field                                                         | Meaning                                                                                                                                                                      |
| ------------ | ------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Time split   | `timing.pct_llm` / `timing.pct_tool`                          | Share of agent-execution wall time attributed to inference-service waits versus the remaining tool-side time. These are wall-time shares, not GPU-active or CPU-active time. |
| Time split   | `timing.avg_wall_sec`                                         | Average agent-execution wall time per trial.                                                                                                                                 |
| Time split   | `timing.avg_llm_sec` / `timing.avg_tool_sec`                  | Average wall time per trial attributed to inference-service waits and tool-side work.                                                                                        |
| Time split   | `timing.avg_llm_calls`                                        | Average number of tracked LLM calls per trial.                                                                                                                               |
| Time split   | `timing.avg_llm_latency_sec`                                  | Average tracked LLM-call latency, calculated as `avg_llm_sec / avg_llm_calls`.                                                                                               |
| Phase        | `phases.avg_env_setup_sec`                                    | Average environment setup time, including container import or image preparation.                                                                                             |
| Phase        | `phases.avg_agent_setup_sec`                                  | Average agent installation and configuration time.                                                                                                                           |
| Phase        | `phases.avg_agent_exec_sec`                                   | Average agent-execution time.                                                                                                                                                |
| Phase        | `phases.avg_verifier_sec`                                     | Average verification and scoring time.                                                                                                                                       |
| CPU          | `cpu.avg_pct` / `cpu.peak_pct`                                | Average and peak host CPU utilization during the selected [active window](#active-sample-filtering). CPU percentage is normalized across all logical cores.                  |
| CPU          | `cpu.avg_cores` / `cpu.p95_cores` / `cpu.peak_cores`          | Mean, P95, and maximum number of logical cores above the [3% utilization threshold](#active-cores).                                                                          |
| CPU          | `cpu.n_cores`                                                 | Number of logical cores visible on the host.                                                                                                                                 |
| CPU          | `cpu.avg_procs` / `cpu.peak_procs`                            | Average and peak workload-process counts detected by the host process collector; primarily meaningful for local container runtimes such as Enroot.                           |
| Disk I/O     | `disk.avg_read_mb_s` / `disk.avg_write_mb_s`                  | Average host read and write throughput during the selected active window.                                                                                                    |
| Disk I/O     | `disk.peak_write_mb_s`                                        | Peak host write throughput during the selected active window.                                                                                                                |
| Results      | `reward.avg`                                                  | Average verifier reward. A value of `0.0` means failure and `1.0` means success when the verifier uses binary rewards; some verifiers provide partial credit.                |
| Results      | `reward.survival_rate`                                        | `n_full / (n_full + n_partial + n_invalid)`: trials that completed within the timeout.                                                                                       |
| Results      | `reward.n_full` / `reward.n_partial` / `reward.n_invalid`     | `n_full` = completed, `n_partial` = timed out, and `n_invalid` = unparseable.                                                                                                |
| Results      | `reward.n_verified`                                           | Trials for which the verifier returned a reward.                                                                                                                             |
| Results      | `reward.n_verifier_errors`                                    | Trials that recorded a verifier or trial-level exception.                                                                                                                    |
| Call latency | `latency.llm.median` / `p95` / `p99` / `max`                  | Distribution of tracked LLM-call latency in seconds.                                                                                                                         |
| Call latency | `latency.tool.median` / `p95` / `p99` / `max`                 | Distribution of tracked tool-call latency in seconds.                                                                                                                        |
| Call latency | `latency.llm.n_calls` / `latency.tool.n_calls`                | Number of calls included in each latency distribution.                                                                                                                       |
| Inference    | `inference.kv_cache.avg_pct` / `peak_pct`                     | Average and peak KV-cache utilization reported by a monitored vLLM endpoint supplied through `--api-base`.                                                                   |
| Inference    | `inference.queue.avg_running` / `peak_running`                | Average and peak number of running requests.                                                                                                                                 |
| Inference    | `inference.queue.avg_waiting` / `peak_waiting`                | Average and peak number of queued requests.                                                                                                                                  |
| Inference    | `inference.throughput.prompt_tok_per_sec` / `gen_tok_per_sec` | Prompt and generation token throughput.                                                                                                                                      |
| Inference    | `inference.latency.avg_ttft_sec`                              | Average time to first token.                                                                                                                                                 |
| Inference    | `inference.latency.avg_tpot_sec`                              | Average time per output token.                                                                                                                                               |
| Inference    | `inference.latency.avg_e2e_sec`                               | Average end-to-end request latency, when exported by the server.                                                                                                             |
| Inference    | `inference.latency.avg_prefill_sec` / `avg_decode_sec`        | Average prefill and decode latency, when exported by the server.                                                                                                             |
| Inference    | `inference.prefix_cache_hit_pct`                              | Prefix-cache hit rate.                                                                                                                                                       |
| Inference    | `inference.preemptions`                                       | KV-cache preemptions during the measurement interval.                                                                                                                        |

The `latency.*` per-call LLM and tool distributions are currently populated only for `claude-code` runs when session JSONL is available. Separately, the aggregate `timing.*` LLM/tool split supports `terminus-2`, `claude-code`, and `mini-swe-agent` using the sources described under [LLM-service and tool-side time](#llm-service-and-tool-side-time).

`inference` is `null` when no usable monitored vLLM samples are available. Within an inference result, latency and cache-hit fields may also be `null` when the endpoint does not expose the required metrics.

## Baseline and system information

TaskTrace-capture records a five-second idle baseline before the sweep and captures system information once.

| Category           | Field                                                               | Meaning                                                                                                                                         |
| ------------------ | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline           | `baseline.avg_cpu_pct`                                              | Average host CPU utilization during the idle baseline.                                                                                          |
| Baseline           | `baseline.avg_cores_active` / `peak_cores_active`                   | Average and peak active-core counts during the idle baseline.                                                                                   |
| Baseline           | `baseline.n_cores`                                                  | Number of logical cores visible during baseline collection.                                                                                     |
| System: CPU        | `system_info.cpu.model`                                             | CPU model.                                                                                                                                      |
| System: CPU        | `system_info.cpu.sockets` / `cores_per_socket` / `threads_per_core` | CPU topology.                                                                                                                                   |
| System: CPU        | `system_info.cpu.max_mhz` / `min_mhz` / `current_mhz_avg`           | Frequency range and current average frequency.                                                                                                  |
| System: CPU        | `system_info.cpu.l1d_cache` / `l1i_cache` / `l2_cache` / `l3_cache` | Cache sizes and instance counts.                                                                                                                |
| System: CPU        | `system_info.cpu.numa_nodes`                                        | NUMA-node-to-CPU mappings.                                                                                                                      |
| System: GPU        | `system_info.gpu.summary[].name`                                    | GPU model.                                                                                                                                      |
| System: GPU        | `system_info.gpu.summary[].vbios` / `driver`                        | VBIOS and driver versions.                                                                                                                      |
| System: GPU        | `system_info.gpu.summary[].memory_mb`                               | GPU memory in MB.                                                                                                                               |
| System: GPU        | `system_info.gpu.nvidia_smi_q`                                      | Raw `nvidia-smi -q` output, including available PCIe, ECC, clock, power, temperature, NVLink, BAR1, compute-mode, and retired-page information. |
| System: Memory     | `system_info.memory.total_gb` / `available_gb`                      | Total and available host memory.                                                                                                                |
| System: OS         | `system_info.os.kernel` / `distro`                                  | Kernel version and operating-system distribution.                                                                                               |
| System: Slurm      | `system_info.slurm.cluster`                                         | `SLURM_CLUSTER_NAME`, falling back to `CLUSTER_NAME` or `CMD_WLM_CLUSTER_NAME` when available.                                                  |
| System: Slurm      | `system_info.slurm.job_id` / `nodelist` / `partition`               | Slurm job placement.                                                                                                                            |
| System: Slurm      | `system_info.slurm.num_nodes` / `gpus_on_node`                      | Allocation size.                                                                                                                                |
| System: InfiniBand | `system_info.infiniband[].hca_id`                                   | Host channel adapter device name.                                                                                                               |
| System: InfiniBand | `system_info.infiniband[].fw_ver` / `board_id`                      | Firmware and board identifiers.                                                                                                                 |
| System: InfiniBand | `system_info.infiniband[].transport` / `node_guid`                  | Transport and node GUID.                                                                                                                        |
| System: Other      | `system_info.virtualization`                                        | `bare-metal` or the detected hypervisor type.                                                                                                   |
| System: Other      | `system_info.versions.<runtime>` / `system_info.versions.harbor`    | Selected container-runtime and Harbor versions.                                                                                                 |
| System: Other      | `system_info.kernel_cmdline`                                        | Kernel boot parameters, including available frequency, IOMMU, and huge-page settings.                                                           |

System-information fields are collected on a best-effort basis and may be absent when the required operating-system command or hardware is unavailable.

## Raw samples in `scaling_results.json`

During a sweep, the default host monitor records a sample every 0.5 seconds:

| Field                                | Meaning                                                                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `t`                                  | Seconds since host sampling began.                                                                                           |
| `cpu_total`                          | Host CPU utilization across all logical cores.                                                                               |
| `cores_active`                       | Number of logical cores above the 3% utilization threshold.                                                                  |
| `n_cores`                            | Number of logical cores visible to the monitor.                                                                              |
| `n_containers`                       | Running containers detected for the selected runtime.                                                                        |
| `n_procs`                            | Workload processes detected by the host process collector; primarily meaningful for local container runtimes such as Enroot. |
| `proc_cats`                          | Counts of workload processes by category, including Python, Git, shell, compiler, curl, package manager, and Node.           |
| `disk_read_mb_s` / `disk_write_mb_s` | Host disk throughput during the sample interval.                                                                             |

**These are host-level samples.** On a shared node, unrelated processes and disk activity can affect the values. Use isolated nodes or controlled allocations when comparing systems.

## How metrics are calculated

### Active-sample filtering

A Harbor run includes environment setup, agent initialization, verification, and teardown in addition to agent execution. TaskTrace-capture selects samples for the agent-execution window using the following fallback order:

1. Read `agent_execution.started_at` and `agent_execution.finished_at` from trial `result.json` files, derive the elapsed span from the earliest agent start to the latest agent finish, and use that span to select the highest-CPU sliding window.
2. If trial timing cannot produce a usable window, select samples where workload processes were detected by the host process collector.
3. If process detection is unavailable, select samples where containers were detected.
4. As a last resort, use all samples.

The CPU and disk summary fields in `report.json` are calculated from the selected samples.

### LLM-service and tool-side time

TaskTrace-capture calculates tool-side time as agent-execution wall time minus tracked LLM-service time. The resulting values describe request-wait time versus the remainder of the agent-execution window; they do not directly measure GPU-active time or CPU-active time.

Timing sources depend on the agent harness:

- `terminus-2` uses `api_request_times_msec` from Harbor trial metadata.
- `claude-code` uses `duration_api_ms` from its stream log and falls back to session timestamps when necessary.
- `mini-swe-agent` derives LLM and tool intervals from timestamped assistant and tool messages in its trajectory.

When no supported timing source is available, the current parser assigns zero LLM-service time and attributes the full agent-execution wall time to the tool-side bucket. This is otherwise unattributed time, not measured tool activity, and should not be interpreted as an LLM/tool-time breakdown.

### Baseline CPU

Before the sweep starts, TaskTrace-capture records a five-second idle CPU baseline. Compare this baseline with the active-window CPU metrics to identify background activity on the host.

### Active cores

For each sample, TaskTrace-capture counts logical cores whose utilization exceeds 3%. On large systems, scheduler placement can spread light work across many cores, so active-core counts should be interpreted together with total CPU utilization.

### Disk I/O

Disk throughput is calculated from changes in Linux block-device counters between samples. It is host-wide rather than container-scoped, so other activity on the node can contribute to the reported values.

## Optional profiling metrics

The default host metrics can be extended with opt-in profiling:

- `--instrument-shell-commands` records command-level timing and produces a Perfetto-compatible timeline.
- `--profile-live-cpu` provides Docker cgroup-v2 CPU accounting.
- `--profile-scheduler required` adds scheduler-based peak-concurrency measurement. It requires `--profile-live-cpu`, `--profile-cpuset-cpus`, concurrency 1, and `--profile-scheduler-collector-image`.
