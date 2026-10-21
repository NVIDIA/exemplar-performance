# Agentic Recipes (TaskTrace capture and replay)

LLMB recipes that benchmark agentic coding workloads: the tool-execution side on NVIDIA Grace CPUs and, in their GPU variants, the model-serving side on the same node. They use the standard install flow: `llmb-install` to provision, `llmb-run submit` to launch a benchmark.

## Introduction

**TaskTrace** is a pair of tools for recording and re-executing agentic coding runs: `tasktrace-capture` runs an agent and records everything it does, and `tasktrace` re-executes that recording and checks that the cluster reproduces it. `tasktrace` can also replay trajectories in ATIF, the [Agent Trajectory Interchange Format](https://harborframework.com/docs/trajectory-format) that Harbor uses to record agent runs; the recipes here replay `tasktrace-capture` recordings. Both are vendored under `inference/tools/` and built from source at install time. These recipes wrap the two tools as llmb-run workloads.

### The agentic flow

An agentic coding run is a loop. A harness (here mini-swe-agent) sends the task and the conversation so far to a model; the model answers with a shell command; the harness executes that command inside the agent's **sandbox** — a **task container**, an isolated container holding the repository under test — and feeds the output back to the model. This repeats until the model declares the task done and a verifier scores the result. Every run therefore spends its time in two places: waiting on the model (**LLM calls**) and running the model's shell commands in the sandbox (**tool calls**).

### What these recipes measure

The CPU recipes qualify the **sandbox side** of agentic workloads on a cluster: does its container path execute an agent's tool calls correctly and reproducibly? They deliberately leave the model out. In the capture recipe the model runs behind an external endpoint; in the replay recipe it is not called at all. The cluster's GPUs are not exercised by them.

The GPU recipes add the **serving side**: a vLLM server on the node's own GPUs answers the agent's calls during a capture, and a replay re-issues every recorded LLM call to that server with the recorded prompt and completion lengths, alongside the tool replay.

## The recipes and their relation

We have two recipe modes: capture + replay and just replay. Each mode can be run in a CPU and a GPU variant. Capture comes first: it produces the trajectory that replay consumes.

1. **Capture + replay** (`inference_tasktrace_capture_replay`) runs the agent live against an external OpenAI-compatible endpoint (the public NVIDIA API by default), records the trajectory, then in the same Slurm job replays that trajectory and checks fidelity. It needs an API key and produces a new trajectory, staged with a provenance record in the layout described below.
2. **Replay** (`inference_tasktrace_replay`) only replays a trajectory that already exists. In this case the recipe provides the trajectory.
3. **Capture + replay on GPUs** (`inference_tasktrace_capture_replay_gpu`) is recipe 1 with a vLLM server on the node's four GPUs in place of the external endpoint; its replay also re-issues the LLM calls. No API key.
4. **Replay on GPUs** (`inference_tasktrace_replay_gpu`) is recipe 2 with the same server, replaying both the tool calls and the LLM calls.

A trajectory has to be captured once, somewhere, before anything can be replayed. After that, anyone can take the trajectory and replay it on another cluster.

## Available recipes

All recipes run on NVIDIA Grace CPUs (Arm Neoverse V2) with enroot, and have been tested only in GB300 and GB200 nodes.

| Workload key                             | GPUs             | What runs                                                                                                                                                   | Path                                              |
| ---------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| `inference_tasktrace_capture_replay`     | none             | Live mini-swe-agent run on `django__django-11490` via an external endpoint, then replay of that trajectory                                                  | `tasktrace-capture-replay/swebench-verified/`     |
| `inference_tasktrace_replay`             | none             | Replay of the shipped SWE-bench Verified `django__django-11490` trajectory                                                                                  | `tasktrace-replay/swebench-verified/`             |
| `inference_tasktrace_capture_replay_gpu` | 4 GB300 or GB200 | vLLM (Nemotron 3 Ultra NVFP4, TP4) on the node; live mini-swe-agent run on `django__django-11490` against it, then replay of both halves of that trajectory | `tasktrace-capture-replay-gpu/swebench-verified/` |
| `inference_tasktrace_replay_gpu`         | 4 GB300 or GB200 | vLLM on the node; replay of both halves of the shipped `django__django-11490` trajectory                                                                    | `tasktrace-replay-gpu/swebench-verified/`         |

The GPU recipes also need about 356 GB of model weights, downloaded once per install.

See the per-recipe `README.md` inside each folder for prerequisites, overrides, outputs and trajectory provenance.

## Viewing results

All four recipes write `summary.json` into the job's experiment directory and share one command to read it. `llmb-run jobs` shows Slurm status only; the verdict command adds the outcome and why.

### One run

```bash
python3 $LLMB_REPO/inference/slurm_recipes/agentic/common/tasktrace_recipe.py verdict /path/to/job/dir
```

`llmb-run jobs log <job_id> --dir` prints a run's directory.

Example output for a passing capture+replay run:

```text
Outcome     : PASS
────────────────────────────────────────────────────────────────────────────────────────────────────
Task        : django__django-11490
Job         : <job id>
Model       : openai/nvidia/nemotron-3-ultra-550b-a55b
Capture     : reward 1.0  |  91 llm calls  |  91/92 tool actions  |  llm wait <sec>  |  tool busy <sec>
Image       : same sha256 in capture and replay
Replay      : 91/92 actions  |  0 mismatches  |  0 timed out  |  pacing unpaced
Tool time   : <ratio> replay/live  (replay <sec>, live <sec>)
Wall time   : <sec> replay  (recorded <sec>)
Summary     : .../summary.json
```

`PASS` means every recorded tool call re-ran with its recorded outcome. An action that the recorded run never executed is skipped, so `91/92 actions` can still pass.

`Tool time` is the replay's tool time divided by the capture's. A capture recorded without timings (`RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS=omit`) has nothing to divide by: the row shows the replay's own time, and `llm wait` and `tool busy` show `n/a`. `reward` is the verifier's score for the agent's answer. Neither affects PASS or FAIL.

The replay-only report omits the `Model`, `Capture`, and `Image` rows, and its `Tool time` row shows only the replay's own time (`<sec> replay`).

The GPU recipes add two rows. `Server` describes the vLLM server the run brought up. `LLM time` is the LLM side's counterpart of `Tool time`: a ratio on a capture+replay with timings, otherwise the replay's own time. For a PASS, every recorded LLM call must replay. Example output for a GPU replay run:

```text
Server      : vllm 0.20.1  |  nemotron-3-ultra-nvfp4  |  TP4  |  4x NVIDIA GB300  |  ready in <sec>
LLM time    : <sec> replay  |  111/111 calls  |  pacing unpaced
```

On `FAIL` the report adds a "Why it failed" section, one root cause per line with follow-on effects attributed to it, and the command exits non-zero.

Causes are named for timeouts (trace-flagged network dependency, recording close to the limit, interactive command after a divergence), for mismatches (environment difference between the live and replay shells, with the live tool output when the capture recipe has it), for the LLM side of a GPU run (calls that failed or could not be replayed), and for launcher aborts before the run (endpoint overloaded or silent, key rejected, model id unknown, too few CPUs, missing image). Anything else is reported as not identifiable rather than guessed.

### Every run under a directory

Point the same command at a directory of runs, such as a workload's `experiments/` tree or the whole `workloads/` folder, and it prints one row per run and exits non-zero if any failed:

```bash
python3 $LLMB_REPO/inference/slurm_recipes/agentic/common/tasktrace_recipe.py verdict $LLMB_INSTALL/workloads
```

Example output:

```text
Job ID    Outcome  Mode            LLM wait  Tool busy  Replay                       Tool time        LLM time  Notes
────────────────────────────────────────────────────────────────────────────────────────────────────
<job id>  FAIL     capture+replay     <sec>      <sec>  128/129 act, 1 mism, 1 t/o  <ratio> (live)               -  the recording already used <n>% of the limit; a slower core or fewer CPUs is enough to exceed it
<job id>  PASS     capture+replay     <sec>      <sec>  91/92 act, 0 mism           <ratio> (live)               -
<job id>  PASS     replay gpu             -          -  149/150 act, 0 mism                      -               -

3 runs: 2 PASS, 1 FAIL
```

- `Outcome` is the PASS/FAIL verdict described above.
- `Mode` is `capture+replay` or `replay`; GPU runs append `gpu` (shown as `replay gpu`).
- `LLM wait` and `Tool busy` are the live run's time waiting on the model and running tool calls; replay-only runs have no live phase, and a capture recorded without timings shows `n/a`.
- `Tool time` is the capture+replay ratio from the single-run report; `-` for replay-only runs and captures recorded without timings.
- `LLM time` is the same ratio for the LLM side of a capture+replay GPU run; `-` otherwise.
- `Notes` is the first diagnosed cause of a failure; empty for a passing run.
- Jobs that are submitted but not started show as `PENDING`, and started jobs without a summary yet as `RUNNING`; neither affects the exit code. For a single run in those states the report says so and exits 2, so a script cannot mistake an unfinished run for PASS (0) or FAIL (1).

`--oneline` prints plain lines for scripts and grep; the same causes are stored as `diagnostics` in `summary.json`.

## Troubleshooting

- **The job runs on nodes that aren't NVIDIA Grace, or is rejected with "Cannot find GPU specification".** The CPU recipes go to the install's CPU partition, which on some clusters holds other Arm CPUs (for example Ampere or Graviton), or to a GPU partition that only accepts jobs that request GPUs. Run on the GB300/GB200 GPU partition instead:
  - For one run: `--slurm-arg partition=<gpu-partition> --slurm-arg gres=gpu:4`.
  - For every run: set the install's CPU partition to the GPU partition with 4 GPUs. Choose it when `llmb-install` asks, set `cpu_partition` and `cpu_partition_gres: 4` in a [headless playfile](../../../cli/llmb-install/docs/headless-installation.md), or change `slurm.cpu.partition` and `slurm.cpu.gres` in an existing `cluster_config.yaml`.
- **The install stops with "Could not detect the NVLink domain"** (or reports a size other than NVL72) on a GB200/GB300 cluster. In an interactive install, choose NVL72 when asked. In a headless install, add `nvl_domain: 72` to the playfile. The CPU recipes use no GPU, so for them the value doesn't matter; it does for the GPU recipes.

## Repository layout

Trajectories are stored under `inference/trajectories/<model>/<agent-harness>/<dataset>/<task>/`, each with a `trace.json` manifest that the launch scripts resolve the `-s` selector against. Two ship today: `nemotron-3-ultra-nvfp4/`, recorded against the public NVIDIA API for the CPU recipes, and `nemotron-3-ultra-vllm/`, recorded by the GPU capture recipe for the GPU recipes. Both were recorded without their timings: each tool call keeps its command, its outcome and the LLM call it followed. The capture recipes stage their output in that same layout with a provenance record, with timings unless `RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS=omit`.
