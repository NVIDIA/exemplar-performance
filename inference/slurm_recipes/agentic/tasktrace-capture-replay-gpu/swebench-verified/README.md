# TaskTrace recipe — GPU live capture and replay of a SWE-bench Verified run (GB300, GB200)

LLMB recipe that runs one live agentic coding task on a GB300 or GB200 node against a vLLM server on the node's own four GPUs, captures the trajectory (every tool call and LLM call the agent made, with timing and outcome), and then replays both halves of that fresh trajectory on the same node: the tool side in the same task container, the LLM side against the same server. No API key and no external endpoint are involved. The recipe checks that the replay reproduces the live run's recorded outcomes and stages the trajectory with a provenance record ready for promotion into `inference/trajectories/`.

## Properties

| Field                   | Value                                                                                                                                                               |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Workload key**        | `inference_tasktrace_capture_replay_gpu`                                                                                                                            |
| **Capture tool**        | `tasktrace-capture` CLI                                                                                                                                             |
| **Replay tools**        | `tasktrace` CLI (tool side), `tasktrace_recipe.py llm-replay` (LLM side)                                                                                            |
| **Task**                | SWE-bench Verified `django__django-11490` (Harbor dataset `swebench-verified`)                                                                                      |
| **Agent**               | mini-swe-agent, built-in default profile (step limit 0, cost limit 3.0)                                                                                             |
| **Model server**        | vLLM, `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`, TP4, served as `nemotron-3-ultra-nvfp4` ([serving config](../../common/serving/nemotron-3-ultra-nvfp4.env)) |
| **Serving image**       | `nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:9fefbd10…` (vLLM 0.20.1; multi-arch, imported by `llmb-install`)                                                      |
| **Task container**      | `ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11490@sha256:58f9edb89a621c2121dc3132ce311130d61d8a3a8075e250231f2a9d757b9a78`                          |
| **GPU**                 | GB300, GB200 (one node, 4 GPUs)                                                                                                                                     |
| **Runtime**             | enroot (server, live task container, replay)                                                                                                                        |
| **Trajectory selector** | `-s nemotron-3-ultra-vllm_miniswe_django-11490` (`<model>_<agent-harness>_<task>`)                                                                                  |

## What it measures

An agent run alternates between waiting on the model (LLM calls) and running tool calls in its sandbox. This recipe runs both halves live once, on one node, then re-runs both from the trajectory it just captured:

1. **Capture** (`tasktrace-capture --instrument-shell-commands`): the live run inside the task image against the node's server, recording per-tool-call wall times, per-LLM-call token counts and wall times, the verifier's reward, host telemetry and the server's metrics.
2. **Replay**: both halves of the new trajectory on the same node, as in the [replay GPU recipe](../../tasktrace-replay-gpu/swebench-verified/README.md#what-it-measures): the tool side at the recorded pacing (`RUN_CONF_TASKTRACE_PACING`), checked against the recorded outcomes, and the LLM side against the same server.

It answers: does this node reproduce a run it just made, on both the sandbox and the serving side?

The live run is nondeterministic: its trajectory, and possibly its reward, will differ from the trajectory shipped with the [replay GPU recipe](../../tasktrace-replay-gpu/swebench-verified/README.md), which this recipe produced.

Commands run with the same container environment and PID isolation as the CPU recipes; see [Container environment and isolation](../../tasktrace-capture-replay/swebench-verified/README.md#container-environment-and-isolation).

## Prerequisites

1. **A Slurm cluster with GB300 or GB200 nodes and enroot.** The job takes one node and its four GPUs from the GPU partition in `cluster_config.yaml`.
2. **Compute-node access to `github.com`, `pypi.org`, `astral.sh` and the Ubuntu package mirror.** The live run fetches the task and its verifier from GitHub and installs the agent harness inside the task container (apt packages, then uv from astral.sh, then the harness from PyPI).

## Prepare environment

Use the **installer** referenced in the [main README](../../../../../README.md) to prepare the recipe environment. The following directory layout and key variables are used:

- `LLMB_INSTALL`: top-level directory for all benchmarking artifacts.
- `LLMB_WORKLOAD`: `${LLMB_INSTALL}/workloads/inference_tasktrace_capture_replay_gpu`; the `tasktrace` binary lands under `bin/`, the capture tool's venv under `venv/`, and the seeded image cache under `enroot-cache/`.
- Weights: `${LLMB_INSTALL}/datasets/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`, a HuggingFace snapshot at revision `183968f8`; a copy from another install can be reused through `RUN_CONF_SERVER_MODEL_DIR`.
- Images under `${LLMB_INSTALL}/images/`: `ai-dynamo+vllm-runtime+sha256-9fefbd10b63c.sqsh` (serving) and `epoch-research+swe-bench.eval.arm64.django__django-11490+sha256-58f9edb89a62.sqsh` (task).

# Run benchmark

## Quickstart

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Submit the benchmark
llmb-run submit -w inference_tasktrace_capture_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4
```

`--scale 4` is the only accepted scale: one GB300 or GB200 node. A run takes 20 to 45 minutes, mostly the live agent run. `launch.sh` requests 2 hours, enough for an agent run that uses its full 40-minute budget.

`RUN_CONF_TASKTRACE_TIMEOUT` sets the replay's per-action timeout in seconds, which otherwise comes from the trajectory (mini-swe-agent's 30 s). Raising it is not a fix for fidelity failures: an action that timed out in the live run counts as a mismatch if the replay lets it finish.

`RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS=omit` records the trajectory without its measured timings, like the shipped one. The replay then runs both sides unpaced in the recorded loop order, and the report shows the replay's own tool and LLM time instead of comparing them with the live run's.

### Additional Slurm parameters

`llmb-run submit` has first-class flags for the common Slurm options and `--slurm-arg` for the rest; they go on the `sbatch` command line, so they override the `#SBATCH` directives in `launch.sh`. `ADDITIONAL_SLURM_PARAMS` is not used by this recipe.

```bash
# a reservation, or a specific node
llmb-run submit -w inference_tasktrace_capture_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4 --reservation my_reservation
llmb-run submit -w inference_tasktrace_capture_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4 --nodelist node001

# any other sbatch option, without the leading dashes (repeatable)
llmb-run submit -w inference_tasktrace_capture_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4 --slurm-arg cpus-per-task=16 --slurm-arg time=02:00:00

# a different partition for one run
llmb-run submit -w inference_tasktrace_capture_replay_gpu -s nemotron-3-ultra-vllm_miniswe_django-11490 --scale 4 --env SBATCH_PARTITION=<partition>
```

## Where outputs land

`llmb-run` pre-creates `$LLMB_EXPERIMENT_DIR` under `${LLMB_WORKLOAD}/experiments/`. The recipe writes:

- `summary.json`: live-run facts, outcome verdict, tool-time and LLM-time comparisons, image identity check, server description, secret-scan result.
- `trajectories/<model>/<agent-harness>/<dataset>/<task>/`: the staged trajectory, its `.sha256`, `trace.json` (with `llm_mode: live-server` and the serving config) and `provenance.json`.
- `server/server.json` and `server/vllm.log`: the server as brought up, and its log across the live run and the replay.
- `capture/`: `tasktrace-capture`'s output tree: `report.json`, `report.txt` (including the LLM-versus-tool time split and the server's metrics sampled during the run), `scaling_results.json`, `telemetry.raw`, and `runs/<job>/<trial>/` with the Harbor trial (`result.json`, `trial.log`, `agent/`, `verifier/`). The trial's `agent/` directory holds the raw shell event log, the finalized trajectory and, unless timings are omitted, its Perfetto timeline. Agent-side installs land here too, so this tree is much larger than a replay-only experiment.
- `llm-replay/llm_replay_report.json` and `llm-replay/llm-replay.log`: the LLM side of the replay, one row per call.
- `profile/pipeline_report.json`: image identity, runtime, tool replay outcome. `profile/replay/replay_report.json`: per-action status, return codes, live and replay wall times, `outcome_matches_source`.
- `secret_scan.json`: files scanned for anything shaped like an API key, and any redactions made.
- `slurm-<jobid>.out`: job log.

## Viewing results

```bash
python3 $LLMB_REPO/inference/slurm_recipes/agentic/common/tasktrace_recipe.py verdict "$(llmb-run jobs log <job_id> --dir)"
```

The report, the per-run table for a whole `experiments/` tree, and what each row and cause means are described once in [Viewing results](../../README.md#viewing-results) in the agentic README.

## Troubleshooting

- Partition, GPU-request and NVLink install problems: see Troubleshooting in the [agentic README](../../README.md#troubleshooting).
- Server and replay errors (`server allocation exposes N GPUs`, `serving image missing`, `model snapshot missing`, `did not become ready`, `warm-up request failed`, `llm calls: …`, `port already in use`): as in the [replay GPU recipe](../../tasktrace-replay-gpu/swebench-verified/README.md#troubleshooting).
- `the server is up but did not answer a chat completion`: the endpoint check after warm-up failed; `server/vllm.log` has the reason.
- `tasktrace-capture missing` or `tasktrace binary missing`: the matching install task did not complete. Check `${LLMB_WORKLOAD}/uv/uv-install.log`, `${LLMB_WORKLOAD}/rust/rustup-install.log`, and re-run `llmb-install`.
- `capture produced no finalized trajectory`: the live run did not reach a finalized trajectory. The job log lists the trial directories and the tail of each `trial.log`; common causes are missing egress to `github.com` (Harbor task fetch) or `pypi.org` (agent install inside the container).
- `No other tasktrace-capture process on this node` failing in preflight: a previous capture is still running on the node; the tool allows one instance per node.
- `image_identity_check.match` false: the replay saw a different `.sqsh` than the capture. Check `${LLMB_WORKLOAD}/enroot-cache/` for a stale import and remove it so the recipe re-seeds the link to the installer's image.
- Outcome `FAIL` with `exit_code_mismatch` on network actions: the compute node could not reach the package index the live run used minutes earlier.
