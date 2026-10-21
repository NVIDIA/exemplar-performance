# TaskTrace recipe — replay of a SWE-bench Verified trajectory (Grace CPU)

LLMB recipe that replays the tool side of a recorded agentic coding run: every tool call in the shipped trajectory is re-executed back to back, without calling an LLM, inside the original SWE-bench task container on an NVIDIA Grace CPU node, and checked against its recorded outcome. No GPU usage.

## Properties

| Field                   | Value                                                                                                                                      |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| **Workload key**        | `inference_tasktrace_replay`                                                                                                               |
| **Tool**                | `tasktrace` CLI                                                                                                                            |
| **Trajectory**          | `inference/trajectories/nemotron-3-ultra-nvfp4/mini-swe-agent/swebench-verified/django__django-11490/`                                     |
| **Task container**      | `ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11490@sha256:58f9edb89a621c2121dc3132ce311130d61d8a3a8075e250231f2a9d757b9a78` |
| **CPU**                 | NVIDIA Grace (Arm Neoverse V2), tested only in GB300 and GB200 nodes                                                                       |
| **Runtime**             | enroot                                                                                                                                     |
| **Trajectory selector** | `-s nemotron-3-ultra_miniswe_django-11490` (`<model>_<agent-harness>_<task>`)                                                              |

## What it measures

An agent run alternates between waiting on the model (LLM calls) and running tool calls in its sandbox. This recipe re-runs only the tool calls, from a trajectory that recorded each one with its exit status, in the order the agent ran them. It answers: does this cluster's sandbox path reproduce the recorded run?

A PASS means this cluster reproduced every recorded tool call.

Commands run with the same container environment and PID isolation as the capture; see [Container environment and isolation](../../tasktrace-capture-replay/swebench-verified/README.md#container-environment-and-isolation).

## Prerequisites

1. **A Slurm cluster with GB300 or GB200 nodes and enroot.** The recipe uses only their Grace CPUs. `llmb-run` submits the job to the CPU partition in `cluster_config.yaml` when one is configured, otherwise to the GPU partition. The launch script refuses a node whose architecture differs from the arm64 task image.

## Prepare environment

Use the **installer** referenced in the [main README](../../../../../README.md) to prepare the recipe environment. The following directory layout and key variables are used:

- `LLMB_INSTALL`: top-level directory for all benchmarking artifacts.
- `LLMB_WORKLOAD`: `${LLMB_INSTALL}/workloads/inference_tasktrace_replay`; the `tasktrace` binary lands under `bin/`.
- Task image: `${LLMB_INSTALL}/images/epoch-research+swe-bench.eval.arm64.django__django-11490+sha256-58f9edb89a62.sqsh`.

# Run benchmark

## Quickstart

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Submit the benchmark
llmb-run submit -w inference_tasktrace_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1
```

`--scale 1` is the only accepted scale and means one node replaying one trajectory. The replay runs the recorded commands back to back without calling an LLM, so the job takes about a minute.

`RUN_CONF_TASKTRACE_TIMEOUT` sets the per-action timeout in seconds, which otherwise comes from the trajectory (mini-swe-agent's 30 s). Raising it is not a fix for fidelity failures: an action that timed out in the recording counts as a mismatch if the replay lets it finish.

### Additional Slurm parameters

`llmb-run submit` has first-class flags for the common Slurm options and `--slurm-arg` for the rest; they go on the `sbatch` command line, so they override the `#SBATCH` directives in `launch.sh`. `ADDITIONAL_SLURM_PARAMS` is not used by this recipe.

```bash
# a reservation, or a specific node
llmb-run submit -w inference_tasktrace_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1 --reservation my_reservation
llmb-run submit -w inference_tasktrace_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1 --nodelist node001

# any other sbatch option, without the leading dashes (repeatable)
llmb-run submit -w inference_tasktrace_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1 --slurm-arg cpus-per-task=16 --slurm-arg time=01:00:00

# a different partition for one run
llmb-run submit -w inference_tasktrace_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1 --env SBATCH_PARTITION=<partition>
```

## Where outputs land

`llmb-run` pre-creates `$LLMB_EXPERIMENT_DIR` under `${LLMB_WORKLOAD}/experiments/`. The recipe writes:

- `summary.json`: outcome verdict, replay summary, and the replay's tool time (`comparison.tool_busy_replay_sec`).
- `profile/pipeline_report.json`: image identity, runtime, replay outcome.
- `profile/replay/replay_report.json`: per-action status, return codes, wall times and `outcome_matches_source`.
- `slurm-<jobid>.out`: job log.

## Viewing results

```bash
python3 $LLMB_REPO/inference/slurm_recipes/agentic/common/tasktrace_recipe.py verdict "$(llmb-run jobs log <job_id> --dir)"
```

The report, the per-run table for a whole `experiments/` tree, and what each row and cause means are described once in [Viewing results](../../README.md#viewing-results) in the agentic README.

## Trajectory provenance

Trajectories live in the shared tree `inference/trajectories/<model>/<agent-harness>/<dataset>/<task>/`, one `trajectory.jsonl` per task with its sha256. This recipe reads `nemotron-3-ultra-nvfp4/mini-swe-agent/swebench-verified/`. Each trajectory directory carries a `trace.json` manifest (selector, tree coordinates, agent, default live model, task image reference and arch) that `launch.sh` resolves the `-s` selector against.

| Field           | Value                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------- |
| Task            | SWE-bench Verified `django__django-11490` (Harbor dataset `swebench-verified`)                  |
| Agent           | mini-swe-agent 2.4.6                                                                            |
| Model           | `nvidia/nemotron-3-ultra-550b-a55b` via the public NVIDIA API                                   |
| Capture tool    | `tasktrace-capture --instrument-shell-commands --source-image-bashrc --trajectory-timings omit` |
| Captured on     | NVIDIA Grace (Arm Neoverse V2) on a GB300 node, enroot, 2026-10-12                              |
| Schema          | `tasktrace-cpu-trace.v1`                                                                        |
| Recorded run    | 120 LLM calls, 121 tool calls                                                                   |
| Recorded reward | 1.0                                                                                             |
| Checksum        | `trajectory.jsonl.sha256` next to the trajectory                                                |

The recorded run has no package or network-dependent tool calls; the replay needs no network.

## Troubleshooting

- Partition, GPU-request and NVLink install problems: see Troubleshooting in the [agentic README](../../README.md#troubleshooting).
- `tasktrace binary missing`: the `install_tasktrace_replay` task did not complete. Check `${LLMB_WORKLOAD}/rust/rustup-install.log` and re-run `llmb-install`.
- `container command completed without a replay report` in `profile/`: the controller binary could not run inside the task image. The install builds a static musl binary for the compute nodes' CPU architecture (`LLMB_NODE_ARCHITECTURE`) to prevent this; check `${LLMB_WORKLOAD}/bin/tasktrace.version` for the target and rebuild if the binary was copied from elsewhere.
- `task image missing`: the installer did not import `container.images`; re-run `llmb-install`, or point `RUN_CONF_IMAGE` at an existing `.sqsh`.
- `allocation exposes N CPUs, need M`: the job got fewer CPUs than `RUN_CONF_TASKTRACE_CPUS` pins. The launch script requests 8 per task (`#SBATCH --cpus-per-task=8`); on a CPU partition that is what you get, on a GPU partition usually more. Raise it with `--slurm-arg cpus-per-task=N` on `llmb-run submit`, or lower `RUN_CONF_TASKTRACE_CPUS`.
- Outcome `FAIL` with `exit_code_mismatch` on network actions: the compute node could not reach the package index the recorded run used.
