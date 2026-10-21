# TaskTrace recipe — GPU replay of a SWE-bench Verified trajectory (GB300, GB200)

LLMB recipe that replays both halves of a recorded agentic coding run on one GB300 or GB200 node. A vLLM server on the node's four GPUs serves the model; the tool side of the trajectory is re-executed inside the original SWE-bench task container as in the [CPU replay recipe](../../tasktrace-replay/swebench-verified/README.md), and the LLM side is replayed against the server, one request per recorded call with the recorded prompt and completion lengths.

## Properties

| Field                   | Value                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Workload key**        | `inference_tasktrace_replay_gpu`                                                                                                                                    |
| **Tools**               | `tasktrace` CLI (tool side), `tasktrace_recipe.py llm-replay` (LLM side)                                                                                            |
| **Trajectory**          | `inference/trajectories/nemotron-3-ultra-vllm/mini-swe-agent/swebench-verified/django__django-11490/` (recorded on GB300 against this server)                       |
| **Model server**        | vLLM, `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`, TP4, served as `nemotron-3-ultra-nvfp4` ([serving config](../../common/serving/nemotron-3-ultra-nvfp4.env)) |
| **Serving image**       | `nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:9fefbd10…` (vLLM 0.20.1; multi-arch, imported by `llmb-install`)                                                      |
| **Task container**      | `ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11490@sha256:58f9edb89a621c2121dc3132ce311130d61d8a3a8075e250231f2a9d757b9a78`                          |
| **GPU**                 | GB300, GB200 (one node, 4 GPUs)                                                                                                                                     |
| **Runtime**             | enroot (server and task container)                                                                                                                                  |
| **Trajectory selector** | `-s nemotron-3-ultra-vllm_miniswe_django-11490` (`<model>_<agent-harness>_<task>`)                                                                                  |

## What it measures

The CPU replay recipe re-runs only an agent run's tool calls. This recipe also re-runs its LLM calls, against the node's own vLLM server:

1. **Tool side**, as in the CPU recipe: every recorded tool call re-executed in the task container and checked against its recorded outcome.
2. **LLM side**: every recorded LLM call re-issued as synthetic tokens of the recorded prompt and completion lengths, so the chat template, tool parser and reasoning parser never run. Each prompt extends the previous one, so the server's prefix cache gets the same reuse as in the live run.

Both sides run together, after warm-up requests that absorb the server's kernel compilation. The shipped trajectory carries no recorded timings, so they replay unpaced in the agent loop's recorded order: a tool call starts once the LLM call it followed has been replayed, and an LLM call once the tool calls before it have run. A trajectory recorded with its timings replays on its recorded schedule instead (`RUN_CONF_TASKTRACE_PACING`, default `recorded`), loading the node as the live run did. It answers: does this node reproduce the recorded run, on both the sandbox and the serving side?

A PASS means every recorded tool call reproduced its recorded outcome and every recorded LLM call replayed.

Commands run with the same container environment and PID isolation as the CPU recipes; see [Container environment and isolation](../../tasktrace-capture-replay/swebench-verified/README.md#container-environment-and-isolation).

## Prerequisites

1. **A Slurm cluster with GB300 or GB200 nodes and enroot.** The job takes one node and its four GPUs from the GPU partition in `cluster_config.yaml`.

## Prepare environment

Use the **installer** referenced in the [main README](../../../../../README.md) to prepare the recipe environment. The following directory layout and key variables are used:

- `LLMB_INSTALL`: top-level directory for all benchmarking artifacts.
- `LLMB_WORKLOAD`: `${LLMB_INSTALL}/workloads/inference_tasktrace_replay_gpu`; the `tasktrace` binary lands under `bin/`.
- Weights: `${LLMB_INSTALL}/datasets/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`, a HuggingFace snapshot at revision `183968f8`; a copy from another install can be reused through `RUN_CONF_SERVER_MODEL_DIR`.
- Images under `${LLMB_INSTALL}/images/`: `ai-dynamo+vllm-runtime+sha256-9fefbd10b63c.sqsh` (serving) and `epoch-research+swe-bench.eval.arm64.django__django-11490+sha256-58f9edb89a62.sqsh` (task).

# Run benchmark

## Quickstart

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Submit the benchmark
llmb-run submit -w inference_tasktrace_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4
```

`--scale 4` is the only accepted scale: one GB300 or GB200 node. A run takes about 20 minutes. `launch.sh` requests 1 hour, enough for a server that needs its full 30-minute start-up allowance to load the weights.

`RUN_CONF_TASKTRACE_TIMEOUT` sets the per-action timeout in seconds, which otherwise comes from the trajectory (mini-swe-agent's 30 s). Raising it is not a fix for fidelity failures: an action that timed out in the recording counts as a mismatch if the replay lets it finish.

### Additional Slurm parameters

`llmb-run submit` has first-class flags for the common Slurm options and `--slurm-arg` for the rest; they go on the `sbatch` command line, so they override the `#SBATCH` directives in `launch.sh`. `ADDITIONAL_SLURM_PARAMS` is not used by this recipe.

```bash
# a reservation, or a specific node
llmb-run submit -w inference_tasktrace_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4 --reservation my_reservation
llmb-run submit -w inference_tasktrace_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4 --nodelist node001

# any other sbatch option, without the leading dashes (repeatable)
llmb-run submit -w inference_tasktrace_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4 --slurm-arg cpus-per-task=16 --slurm-arg time=01:00:00

# a different partition for one run
llmb-run submit -w inference_tasktrace_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4 --env SBATCH_PARTITION=<partition>
```

## Where outputs land

`llmb-run` pre-creates `$LLMB_EXPERIMENT_DIR` under `${LLMB_WORKLOAD}/experiments/`. The recipe writes:

- `summary.json`: outcome verdict, both replay summaries, the replay's tool time and LLM time (`comparison`, `llm_comparison`), and the server description.
- `server/server.json` and `server/vllm.log`: the server as brought up, and its log.
- `llm-replay/llm_replay_report.json`: one row per LLM call; `llm-replay/llm-replay.log`: its progress log.
- `profile/pipeline_report.json`: image identity, runtime, tool replay outcome. `profile/replay/replay_report.json`: per-action status, return codes, wall times and `outcome_matches_source`.
- `slurm-<jobid>.out`: job log, including the server bring-up and warm-up.

## Viewing results

```bash
python3 $LLMB_REPO/inference/slurm_recipes/agentic/common/tasktrace_recipe.py verdict "$(llmb-run jobs log <job_id> --dir)"
```

The report, the per-run table for a whole `experiments/` tree, and what each row and cause means are described once in [Viewing results](../../README.md#viewing-results) in the agentic README.

## Trajectory provenance

Trajectories live in the shared tree `inference/trajectories/<model>/<agent-harness>/<dataset>/<task>/`, one `trajectory.jsonl` per task with its sha256. This recipe reads `nemotron-3-ultra-vllm/mini-swe-agent/swebench-verified/`, recorded by the [capture GPU recipe](../../tasktrace-capture-replay-gpu/swebench-verified/README.md) against this same server. Its `trace.json` manifest, which `launch.sh` resolves the `-s` selector against, also names the serving config.

| Field           | Value                                                                                                           |
| --------------- | --------------------------------------------------------------------------------------------------------------- |
| Task            | SWE-bench Verified `django__django-11490` (Harbor dataset `swebench-verified`)                                  |
| Agent           | mini-swe-agent 2.4.6, built-in default profile                                                                  |
| Model           | `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` @ `183968f8`, vLLM 0.20.1, TP4 on 4x GB300                     |
| Capture tool    | `tasktrace-capture --instrument-shell-commands --source-image-bashrc --trajectory-timings omit` (Harbor 0.13.2) |
| Captured on     | NVIDIA Grace on a GB300 node, enroot, 2026-10-12, by the capture GPU recipe                                     |
| Schema          | `tasktrace-cpu-trace.v1`                                                                                        |
| Recorded run    | 111 LLM calls, 112 tool calls                                                                                   |
| Recorded reward | 1.0 (the agent solved the task)                                                                                 |
| Checksum        | `trajectory.jsonl.sha256` next to the trajectory (`4a02b654…`)                                                  |

The recorded run has no package or network-dependent tool calls; the replay needs no network.

## Troubleshooting

- Partition, GPU-request and NVLink install problems: see Troubleshooting in the [agentic README](../../README.md#troubleshooting).
- `server allocation exposes N GPUs, need 4`: the job did not get the node's GPUs. Check `slurm.gpu` in `cluster_config.yaml` (partition and GPUs per node) and submit with `--scale 4`.
- `server serving image missing` or `server model snapshot missing`: the installer did not import the serving image or download the weights; re-run `llmb-install` for this workload, or point `RUN_CONF_SERVER_IMAGE` / `RUN_CONF_SERVER_MODEL_DIR` at existing copies.
- `server did not become ready within 1800s` or `server exited before it became ready`: read `server/vllm.log` in the run directory. A slow filesystem needs a higher `RUN_CONF_SERVER_STARTUP_TIMEOUT`; an engine that dies at start-up usually reports the flag or the memory it could not satisfy.
- `warm-up request of N prompt tokens failed`: the server answered `/health` but could not complete a request of that size; `server/vllm.log` has the engine's stack trace. The run stops before anything is timed.
- `llm calls: N of the replayed LLM calls failed`: see the failed rows in `llm-replay/llm_replay_report.json` (an `error` field each).
- `server port 8000 on 127.0.0.1 is already in use`: another process holds the port on the node; set `RUN_CONF_SERVER_PORT`.
- `the trajectory carries no recorded timings, so it can only replay unpaced`: `RUN_CONF_TASKTRACE_PACING` or `RUN_CONF_LLM_PACING` is set to `recorded`; unset it.
- `tasktrace binary missing`, `task image missing`, `allocation exposes N CPUs, need M`, `exit_code_mismatch` on network actions: as in the [CPU replay recipe](../../tasktrace-replay/swebench-verified/README.md#troubleshooting). Here the launch script requests 32 CPUs per task (128 per node), of which the tool replay pins `RUN_CONF_TASKTRACE_CPUS` and the server takes the rest.
