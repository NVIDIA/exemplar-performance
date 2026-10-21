# TaskTrace-capture

With agentic workloads, performance is now a whole-system problem, not just token throughput. TaskTrace-capture is an end-to-end agent benchmarking harness that runs real agentic tasks, captures system performance and task outcomes, and can emit command-level traces for supported instrumented runs.

TaskTrace-capture enables:

- inference providers to understand their infrastructure against agent workloads
- frontier labs to optimize sandbox infrastructure
- enterprises to size CPU / GPU serving capacity for agentic workloads
- model builders to measure and optimize their model's tool-calling patterns

## Ways to use this tool

Use TaskTrace-capture to:

1. **Profile existing evaluations:** Measure execution time, host CPU and disk activity, and optional vLLM server metrics while running Harbor datasets.

2. **Characterize agentic infrastructure:** Run concurrency sweeps to identify CPU, sandbox, storage, or inference-serving bottlenecks.

3. **Trace tool workloads:** Opt in to command-level timing and produce JSONL and Perfetto artifacts for offline analysis.

## Overview

Agent performance depends on both the sandbox and the inference service. A chat request primarily exercises the inference service, while an agent alternates between model requests and tool execution in the sandbox. CPU, storage, networking, and inference capacity can therefore affect speed, cost, and task outcomes.

Agentic teams need to answer:

- Where does agent wall-clock time go — tool execution or LLM inference?
- How does throughput per CPU core change as I add more concurrent agents?
- How do CPUs compare on identical, reproducible agentic work?
- How does inference-service time compare with tool-side time across tasks, models, and agent harnesses?

TaskTrace-capture is a benchmark harness that orchestrates concurrent agent runs, samples host CPU and disk activity, and generates performance reports. Each run combines a task or dataset, an agent harness, a model, and a container runtime. Users can run registered Harbor datasets, individual Harbor tasks, Harbor-compatible local tasks, or ad hoc prompts.

```text
  ┌────────────────────────────────────────────────────────────────────┐
  │                   TASKTRACE-CAPTURE                                │
  │                                                                    │
  │  WORKLOAD                     EXECUTION CONFIGURATION              │
  │  • Harbor dataset             • Agent harness                      │
  │  • Individual task            • Model endpoint                     │
  │  • Local task                 • Container runtime                  │
  │  • Ad hoc prompt              • Concurrency levels                 │
  │            │                            │                          │
  │            └──────────────┬─────────────┘                          │
  │                           ▼                                        │
  │  Live agent execution with host sampling and optional vLLM metrics │
  │                           │                                        │
  │                           ▼                                        │
  │  Reports · task outcomes · timing · resource data · trajectories   │
  └────────────────────────────────────────────────────────────────────┘

```

## Supported Configurations

| Layer                 | Options                                                                                                                                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agent harnesses**   | Claude Code <br>terminus-2 <br>mini-swe-agent                                                                                                                                                                        |
| **Benchmarks**        | Terminal-Bench 2.0 <br>Terminal-Bench 1.0 <br>SWE-bench Pro <br>SWE-bench Verified <br>MedAgentBench <br>LawBench <br> [Full Harbor registry](https://github.com/laude-institute/harbor-datasets/tree/main/datasets) |
| **Inference**         | **Local**: vLLM, SGLang, TRT-LLM, Dynamo <br>**Cloud APIs**: OpenAI-compatible, Anthropic-compatible                                                                                                                 |
| **Hardware**          | x86 <br>ARM <br>Validated on NVIDIA Grace and x86 Slurm clusters                                                                                                                                                     |
| **Container runtime** | Enroot <br>Docker <br>Apptainer <br>OpenSandbox <br>Current container (in place, e.g. a Kubernetes pod)                                                                                                              |

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- **A container runtime** — one of:
  - [Enroot](https://github.com/NVIDIA/enroot) (default, rootless, pre-installed on NVIDIA HPC clusters) — use `--runtime enroot`
  - Docker (workstations, CI, local dev) — use `--runtime docker`. Requires a reachable Docker daemon. GPU tasks are not yet supported with Docker.
  - [OpenSandbox](https://github.com/alibaba/OpenSandbox) (remote Kubernetes-backed sandboxes) — use `--runtime opensandbox`. Requires a reachable OpenSandbox controller and API key.
  - [Apptainer](https://apptainer.org/) (rootless HPC) — use `--runtime apptainer`. TaskTrace-capture installs and selects a managed userspace Apptainer 1.5.2 by default; no system installation or sudo access is required.
  - The current container (TaskTrace-capture runs inside the task image, for example in a Kubernetes pod) — use `--runtime current`.
- An API key and endpoint URL from a compatible inference provider, or access to a self-hosted OpenAI-compatible inference server

> **Note:** Claude Code agent uses pinned version **2.1.121** (cached automatically on first run; override per-run with `--claude-code-version X.Y.Z` or edit `DEFAULT_CLAUDE_CODE_VERSION` in `tasktrace_capture/runtimes/_shared.py`)

## Quick Start

```bash
git clone https://github.com/NVIDIA/exemplar-performance.git
cd exemplar-performance/inference/tools/tasktrace-capture
uv sync
```

### Run `hello-world`

Run this Quick Start on an Enroot-enabled Slurm compute node, not a login node. TaskTrace-capture launches containers and samples host `/proc`; choose the command that matches your endpoint and agent. The TaskTrace agentic recipes ([Slurm](../../slurm_recipes/agentic/README.md), [Kubernetes](../../k8s_recipes/recipes/agentic/)) set the agent themselves (mini-swe-agent), so this choice only applies when you run TaskTrace-capture on its own.

**Anthropic-compatible endpoint with Claude Code:**

```bash
export ANTHROPIC_BASE_URL="<your-inference-endpoint>"
export ANTHROPIC_API_KEY="<your-api-key>"

uv run tasktrace-capture \
    --agent claude-code \
    -m azure/anthropic/claude-sonnet-4-5 \
    -d hello-world \
    --concurrency 1 \
    --agent-env "ANTHROPIC_BASE_URL=$ANTHROPIC_BASE_URL" \
    --agent-env "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY"
```

**OpenAI-compatible endpoint with mini-swe-agent:**

```bash
export OPENAI_API_BASE="<your-inference-endpoint>/v1"
export OPENAI_API_KEY="<your-api-key>"

uv run tasktrace-capture \
    --agent mini-swe-agent \
    -m "openai/<provider>/<your-model>" \
    -d hello-world \
    --concurrency 1 \
    --agent-env "OPENAI_API_BASE=$OPENAI_API_BASE" \
    --agent-env "OPENAI_API_KEY=$OPENAI_API_KEY"
```

## Runtime options

### Enroot

The Quick Start command uses Enroot, TaskTrace-capture's default runtime, so `--runtime enroot` is optional.

### Docker

On a machine with a reachable Docker daemon, pass `--runtime docker`:

```bash
uv run tasktrace-capture \
    --runtime docker \
    --agent claude-code \
    -m azure/anthropic/claude-sonnet-4-5 \
    -d hello-world \
    --concurrency 1 \
    --agent-env "ANTHROPIC_BASE_URL=<your-inference-endpoint>" \
    --agent-env "ANTHROPIC_API_KEY=<your-api-key>"
```

The Docker path uses the same workload, agent, model, and output format as the default Enroot path. Docker does not currently support tasks that request GPUs. Preflight also requires permission to create and remove a disposable bridge network.

### OpenSandbox (remote Kubernetes sandboxes)

OpenSandbox is opt-in and creates one remote sandbox for each Harbor trial.

```bash
export OPENSANDBOX_URL=https://sandbox.example.com
export OPENSANDBOX_API_KEY="<controller-api-key>"

uv run tasktrace-capture \
    --runtime opensandbox \
    --agent claude-code \
    -m azure/anthropic/claude-sonnet-4-5 \
    --prompt "Create /app/hello.txt containing hello." \
    --concurrency 1 \
    --agent-env "ANTHROPIC_BASE_URL=<your-inference-endpoint>" \
    --agent-env "ANTHROPIC_API_KEY=<your-api-key>"
```

OpenSandbox requires a prebuilt Linux OCI image. GPU tasks, Docker Compose, forced builds, and Dockerfile build steps such as `RUN` and `COPY` are not currently supported. Sandboxes are automatically cleaned up after each trial.

### Apptainer (Docker-free HPC)

Apptainer is opt-in; Enroot remains the default. Apptainer requires a `.sif` file whose architecture matches the compute node. TaskTrace-capture selects its managed userspace Apptainer automatically, but it does not pull or build the evaluation image.

Run from a compute allocation that supports direct SIF mounting:

```bash
uv run tasktrace-capture \
    --runtime apptainer \
    --agent mini-swe-agent \
    -m "openai/<provider>/<your-model>" \
    --prompt "Create /tmp/hello.txt containing hello." \
    --prompt-image /shared/sifs/python-3.12-arm64.sif \
    --concurrency 1 \
    --agent-env "OPENAI_API_BASE=<your-inference-endpoint>/v1" \
    --agent-env "OPENAI_API_KEY=<your-api-key>"
```

Before starting a trial or making a model request, TaskTrace-capture checks the managed Apptainer executable, image architecture, `squashfuse_ll`, and access to `/dev/fuse`.

Normal runs perform these preflight checks automatically. Use `--check` to validate the runtime and exit without starting trials.

TaskTrace-capture's Apptainer runtime accepts only architecture-native `.sif` files. It does not pull OCI images, convert images, replay Dockerfiles, or emulate another CPU architecture. If a prepared SIF is unavailable, use `--runtime enroot` for that workload.

By default, TaskTrace-capture ignores any `apptainer` found on `PATH`, including a Conda installation, and selects its versioned managed runtime. It does not modify or remove the external installation. To deliberately use your own installation:

```bash
export TASKTRACE_CAPTURE_APPTAINER_BIN=/opt/apptainer/bin/apptainer
```

An invalid explicit path is a hard error; TaskTrace-capture never silently falls back to another binary.

### Current container (Kubernetes pods)

Use `--runtime current` when TaskTrace-capture already runs inside the task's container, for example in a Kubernetes pod created from the task image. The trial runs in place.

```bash
export TASKTRACE_CAPTURE_CURRENT_IMAGE_REF="<registry>/<task-image>:<tag>@sha256:<digest>"

uv run tasktrace-capture \
    --runtime current \
    --task <dataset>/<task> \
    --agent mini-swe-agent \
    -m "openai/<provider>/<your-model>" \
    --concurrency 1 \
    --agent-env "OPENAI_API_BASE=$OPENAI_API_BASE" \
    --agent-env "OPENAI_API_KEY=$OPENAI_API_KEY"
```

Set `TASKTRACE_CAPTURE_CURRENT_IMAGE_REF` to the image the container was created from; the trial is refused when it does not match the task's image. Concurrency must be 1; run more trials in more containers. Agent commands start from the container's own environment, not TaskTrace-capture's. `/logs/agent`, `/logs/verifier` and `/logs/artifacts` must not already contain files. GPU tasks are not currently supported.

## Output folder structure

Results are written to `jobs/<timestamp>__sweep__<name>/`:

```text
jobs/<timestamp>__sweep__<name>/
├── scaling_results.json   # Detailed results with raw measurement samples
├── report.json            # Structured summary and system information
├── report.txt             # Human-readable report
└── runs/                  # Harbor job and trial artifacts
    └── <harbor-job-timestamp>/
        ├── config.json
        ├── job.log
        ├── lock.json
        ├── result.json
        └── <task-name>__<trial-id>/
            ├── agent/
            ├── artifacts/
            ├── config.json
            ├── result.json
            ├── trial.log
            └── verifier/
```

Use `report.txt` for a quick review, `report.json` for programmatic analysis, and `scaling_results.json` when raw measurement samples are needed.

<details>
<summary>Example `report.txt`</summary>

```text
Time Split  (% wall time: tool calls on client vs LLM inference on server)
  Concurrency   Tool%    LLM%     Tool      LLM     Wall   Calls  Avg lat
            1   34.2%   65.8%    25.6s    49.3s    74.9s      20    2.47s
            8   41.3%   58.7%    31.2s    44.4s    75.6s      18    2.47s
            32  68.1%   31.9%    89.4s    41.9s   131.3s      15    2.79s

Phase Breakdown  (avg seconds per trial)
  Concurrency  Env setup  Agent setup  Execution  Verifier    Total
            1       0.5s        12.3s      74.9s      6.2s    93.9s
            8       0.5s        24.0s      75.6s      6.8s   106.9s
            32      0.5s        45.1s     131.3s      8.7s   185.6s

CPU Usage  (during agent execution only)
  Concurrency  Avg CPU%  Peak CPU%  Avg cores  P95 cores  Peak cores
            1      0.6%       1.8%        6.2         15      27/128
            8      4.6%      11.8%       35.4         68     102/128
            32     14.2%     46.2%       81.1        114     128/128

Disk I/O  (during benchmark)
  Concurrency  Avg read MB/s  Avg write MB/s  Peak write MB/s
            1            0.1             0.4             12.0
            8            1.5             8.2            120.0
            32           4.0            20.0            350.0

Rewards
  Concurrency  Avg reward    Verified   Completed
            1       1.000         1/1         1/1
            8       0.875         7/8         8/8
            32      0.375       12/32       15/32
```

Runs against a monitored self-hosted vLLM endpoint may also include an Inference Server section with throughput, queue, KV-cache, latency, cache-hit, and preemption metrics.

</details>

## Usage examples

### Docker smoke test

This example uses an OpenAI-compatible cloud endpoint. For `mini-swe-agent`, pass the API base ending in `/v1`; LiteLLM appends the chat-completions route.

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_API_BASE="<your-inference-endpoint>/v1"

uv run tasktrace-capture \
    --runtime docker \
    --agent mini-swe-agent \
    -m "openai/<provider>/<your-model>" \
    -d hello-world \
    --concurrency 1 \
    --agent-env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    --agent-env "OPENAI_API_BASE=$OPENAI_API_BASE" \
    --agent-kwarg "config_file=<path-to-mini-swe-config>.yaml"
```

Important details:

- `mini-swe-agent` routes models through LiteLLM. Use the model prefix required by your provider, such as `openai/<provider>/<your-model>` for an OpenAI-compatible endpoint.
- Supply a mini-swe-agent configuration file appropriate for your model and token limits.
- Keep the endpoint in `OPENAI_API_BASE` for this cloud workflow. Do not pass it as tasktrace-capture's `--api-base`; that option activates self-hosted inference-server endpoint/metrics checks.
- The report's `API` field is blank in this mode because the cloud endpoint is passed directly to mini-swe-agent rather than monitored by TaskTrace-capture.

Preflight and troubleshooting for this Docker smoke test:

- TaskTrace-capture preflight creates and removes a disposable Docker bridge network before starting Harbor. If preflight reports that `docker bridge network creation` failed, restart Docker and rerun the check:

```bash
sudo systemctl restart docker
uv run tasktrace-capture \
    --runtime docker \
    --agent mini-swe-agent \
    --check
```

### Harbor tasks

Use `--task` to run an individual Harbor task:

```bash
uv run tasktrace-capture \
    --agent claude-code \
    -m azure/anthropic/claude-sonnet-4-5 \
    --task terminal-bench/password-recovery \
    --concurrency 1,2,4 \
    --agent-env "ANTHROPIC_BASE_URL=<your-inference-endpoint>" \
    --agent-env "ANTHROPIC_API_KEY=<your-api-key>"
```

To run an explicit suite, provide comma-separated task names from the same dataset:

```bash
uv run tasktrace-capture \
    --agent claude-code \
    -m azure/anthropic/claude-sonnet-4-5 \
    --task terminal-bench/mailman,build-pmars,password-recovery \
    --concurrency 1 \
    --agent-env "ANTHROPIC_BASE_URL=<your-inference-endpoint>" \
    --agent-env "ANTHROPIC_API_KEY=<your-api-key>"
```

For a suite, `--concurrency 1` means one copy of each named task. For a single task, increasing `--concurrency` runs additional copies of that task. In dataset mode, `--n-tasks` selects distinct tasks and `--concurrency` sets the maximum number of active trials.

To select several distinct tasks from a dataset:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_API_BASE="<your-inference-endpoint>/v1"

uv run tasktrace-capture \
    --runtime docker \
    --agent mini-swe-agent \
    -m "openai/<provider>/<your-model>" \
    -d compilebench \
    --n-tasks 4 \
    --trials 1 \
    --concurrency 4 \
    --agent-env "OPENAI_API_KEY=$OPENAI_API_KEY" \
    --agent-env "OPENAI_API_BASE=$OPENAI_API_BASE" \
    --agent-kwarg "config_file=<path-to-mini-swe-config>.yaml"
```

Browse published datasets in [Harbor Hub](https://hub.harborframework.com/datasets).

### SWE-bench Pro

TaskTrace-capture also supports [SWE-bench Pro](https://labs.scale.com/leaderboard/swe_bench_pro_public) through Harbor's dataset registry (731 real-world debugging tasks across Python, Go, JavaScript, and TypeScript codebases).

```bash
# Run one SWE-bench Pro task
uv run tasktrace-capture \
    --agent claude-code \
    -m azure/anthropic/claude-sonnet-4-5 \
    --task "swebenchpro/instance_qutebrowser__qutebrowser-0833b5f6f140d04200ec91605f88704dd18e2970-v059c6fdc75567943479b23ebca7c07b5e9a7f34c" \
    --concurrency 1 \
    --agent-env "ANTHROPIC_BASE_URL=<your-inference-endpoint>" \
    --agent-env "ANTHROPIC_API_KEY=<your-api-key>"
```

Browse tasks at [swebenchpro/1.0](https://github.com/harbor-framework/harbor-datasets/tree/main/datasets/swebenchpro). Use the full task ID including the `-v...` version suffix — truncated IDs will fail.

### Ad hoc prompts (no Harbor task directory)

Use `--prompt` or `--prompt-file` for a one-off workload that is not registered in Harbor. TaskTrace-capture creates a Harbor-compatible task under the sweep output directory and produces the standard reports and supported metrics. It uses the same telemetry pipeline as registered Harbor tasks; available fields still depend on the selected agent and runtime.

```bash
# (i) Default ubuntu:24.04 image, claude-code agent, single prompt at concurrency 1
uv run tasktrace-capture \
    --runtime docker \
    --agent claude-code \
    -m azure/anthropic/claude-sonnet-4-5 \
    --prompt "Create hello.txt containing Hello, world!" \
    --concurrency 1 \
    --agent-env "ANTHROPIC_BASE_URL=<your-inference-endpoint>" \
    --agent-env "ANTHROPIC_API_KEY=<your-api-key>"

# (ii) Custom image + prompt-file + custom bench name + claude-code
uv run tasktrace-capture \
    --runtime docker \
    --agent claude-code \
    -m azure/anthropic/claude-sonnet-4-5 \
    --prompt-file ./my-task.md \
    --prompt-image "<registry>/<image>:<tag>" \
    --prompt-name my-bench \
    --concurrency 1 \
    --agent-env "ANTHROPIC_BASE_URL=<your-inference-endpoint>" \
    --agent-env "ANTHROPIC_API_KEY=<your-api-key>"
```

Available options:

- `--prompt TEXT` — provide the prompt directly
- `--prompt-file PATH` — read the prompt from a file
- `--prompt-image IMAGE` — select the task image; defaults to `ubuntu:24.04`
- `--prompt-name NAME` — select the sweep directory label; defaults to
  `tasktrace-capture-adhoc`

`--prompt` and `--prompt-file` are mutually exclusive. Neither can be combined with `--task` or `--dataset`.

Caveats:

- The generated `tests/test.sh` always writes `1` to `/logs/verifier/reward.txt`, so reward is **always 1.0** regardless of whether the agent actually completed the task. Use ad hoc prompts for telemetry-only sweeps; for pass/fail evaluation, write a real Harbor task directory with a meaningful verifier.
- One prompt per invocation. Batching multiple prompts into a single sweep is not implemented.
- Claude Code tool behavior depends on the model endpoint. Claude Code 2.1.121 with `azure/anthropic/claude-sonnet-4-5` was validated on June 15, 2026: it executed Bash tools without permission prompts under `--permission-mode=bypassPermissions` and passed a meaningful verifier. Validate other endpoint/model combinations before a large sweep; mini-swe-agent remains the simpler OpenAI-compatible harness.

### Self-hosted vLLM or another OpenAI-compatible backend

Use `--api-base` to point TaskTrace-capture at a reachable self-hosted, OpenAI-compatible endpoint. Because `terminus-2` is the default agent, this example does not need `--agent`. TaskTrace-capture validates the endpoint and collects compatible vLLM metrics when they are available.

```bash
uv run tasktrace-capture \
    --task terminal-bench/password-recovery \
    --concurrency 1,4,8,16,32 \
    --api-base http://<host>:8000/v1
```

## Repeatable Runs

For repeatable tests, put all experiment-definition options in one YAML file and check it into source control.

**YAML mode is exclusive for experiment-definition options.** When you pass `-c file.yaml`, TaskTrace-capture uses only the YAML experiment definition. Mixing experiment-definition CLI flags with `--config` is a hard error, while the operational flags `--skip-preflight`, `--check`, and `--dry-run` remain available. This keeps experiments reproducible and removes ambiguity around per-key overrides.

```yaml
# exp.yaml
runtime: docker
agent: claude-code
model: azure/anthropic/claude-sonnet-4-5
dataset: hello-world
concurrency: [1, 4, 16]
agent_env:
  ANTHROPIC_BASE_URL: <your-inference-endpoint>
  ANTHROPIC_API_KEY: ${INFERENCE_API_KEY}
skip_preflight: true
```

```bash
# Works: yaml is the entire experiment definition.
uv run tasktrace-capture -c exp.yaml
# Errors: cannot mix --config with experiment-definition flags.
uv run tasktrace-capture -c exp.yaml --concurrency 4
# tasktrace-capture: error: --config: cannot mix yaml with CLI flags.
#   Saw: --concurrency 4. Edit the yaml or drop --config.
```

`${VAR}` and `$VAR` placeholders in string values are resolved against the process environment at load time; an unset reference is an error.

### Ad hoc prompts via YAML

The `--prompt` / `--prompt-file` / `--prompt-image` / `--prompt-name` flags also work as YAML keys, which is useful when iterating on a custom prompt without retyping the options for every run.

```yaml
# adhoc.yaml
agent: mini-swe-agent
runtime: docker
concurrency: [1, 4]
prompt: "Create /app/hello.txt containing 'Hello, world!'"
prompt_image: ubuntu:24.04
prompt_name: my-bench
model: openai/model
api_base: http://<vllm-host>:8000/v1
agent_env:
  OPENAI_API_KEY: dummy  # vLLM ignores the key, but the client requires a value
  OPENAI_API_BASE: http://<vllm-host>:8000/v1
```

```bash
uv run tasktrace-capture -c adhoc.yaml
```

Ad hoc prompts use a no-op verifier, so their reward is always `1.0`. Use them for performance measurements; for correctness evaluation, create a Harbor task with a meaningful verifier. Only one prompt can be defined per experiment.

## CLI Reference

| Flag                             | Description                                                                                          |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `--agent` / `-a`                 | Harbor agent name (default: `terminus-2`); see [Supported Configurations](#supported-configurations) |
| `--api-base URL`                 | OpenAI-compatible endpoint (required for terminus, or set `AGENT_API_BASE` environment variable)     |
| `--dataset` / `-d`               | Harbor dataset (for example, `hello-world` or `terminal-bench@2.0`)                                  |
| `--n-tasks` / `-l`               | Maximum distinct tasks selected in dataset mode                                                      |
| `--task` / `-p`                  | Single task or comma-separated suite (`terminal-bench/qemu-startup`)                                 |
| `--model` / `-m`                 | Model name (default: `hosted_vllm/model`)                                                            |
| `--concurrency 1,2,4`            | Comma-separated concurrency levels to sweep                                                          |
| `-k` / `--trials N`              | Number of trials per task                                                                            |
| `--repeat N`                     | Re-run each concurrency level N times                                                                |
| `--agent-timeout N`              | Per-agent timeout in seconds (default: 3600)                                                         |
| `--agent-env K=V`                | Environment variables for the agent container (repeatable)                                           |
| `--agent-kwarg K=V`              | Extra keyword arguments passed to the Harbor agent (repeatable)                                      |
| `--output-dir DIR`               | Where to write results (default: `./jobs/<timestamp>__sweep__<name>/`)                               |
| `--runtime NAME`                 | Container runtime: `enroot` (default), `docker`, `apptainer`, `opensandbox`, or `current`            |
| `--allow-parallel-docker`        | Allow multiple isolated Docker experiments on one node                                               |
| `--swebench-image-prefix PREFIX` | Map SWE-bench task IDs to OCI tags for Enroot or Docker                                              |
| `--swebench-sif-template PATH`   | Existing SIF template for Apptainer; supports `{instance_id}`                                        |
| `--instrument-shell-commands`    | Record per-command Bash timing for `terminus`, `claude-code`, and `mini-swe-agent`                   |
| `--source-image-bashrc`          | Set `BASH_ENV=/root/.bashrc` in the container; recorded in the trace and restored by the replay      |
| `--trajectory-timings omit`      | Write the instrumented trajectory without measured timings or a Perfetto timeline (default: `keep`)  |
| `--no-start-barrier`             | Let concurrent tasks start naturally instead of synchronizing agent launch                           |
| `--check`                        | Run preflight checks only; exit 0/1                                                                  |
| `--skip-preflight`               | Skip preflight checks                                                                                |
| `--quiet` / `-q`                 | Suppress preflight output                                                                            |
| `--dry-run`                      | Show the experiment plan without running                                                             |

## Metrics Collected

Every sweep writes structured metrics to `report.json`, a human-readable summary to `report.txt`, and raw samples to `scaling_results.json`.

TaskTrace-capture collects:

- **Results:** reward, completion status, and survival rate
- **Timing:** environment setup, agent setup, task execution, verification, and LLM-versus-tool time for `terminus-2`, `claude-code`, and `mini-swe-agent`
- **Host resources:** CPU utilization, active cores, process counts, and disk I/O
- **Inference server:** optional vLLM metrics, including throughput, queue depth, KV-cache utilization, TTFT, TPOT, cache hits, and preemptions
- **System information:** CPU, GPU, memory, OS, Slurm allocation, container runtime, and relevant software versions

See [docs/metrics.md](./docs/metrics.md) for field definitions, calculation methodology, sampling, and filtering behavior.

## Optional Profiling

### Bash command timing

Add `--instrument-shell-commands` to record individual Bash-command timing and generate a Perfetto-compatible timeline for `terminus`, `claude-code`, and `mini-swe-agent` runs.

Add `--trajectory-timings omit` as well for a trajectory you plan to share without its measurements. The finalized trajectory then has no start, finish, or wall time on any record, and each command records the LLM call it followed instead, which is enough for `tasktrace` to replay it in order. No Perfetto timeline is written, and the run's own reports still measure time as usual.

### Live CPU and peak-concurrency profiling

Docker runs can optionally collect task-scoped CPU usage from cgroup v2 with `--profile-live-cpu`. Peak simultaneous CPU profiling is available with `--profile-scheduler required`. It requires:

- Docker with `--profile-live-cpu`
- an explicit `--profile-cpuset-cpus <cpuset>`
- `--concurrency 1`
- `--profile-scheduler-collector-image <image>`, containing compatible `/usr/bin/perf` and BCC tools for the host kernel

If `--override-cpus` is also supplied, its value must match the CPU-set width.

## Key components

```text
tasktrace_capture/
  cli.py               # Entry point: preflight + sweep orchestration
  runner.py            # Spawns Harbor subprocess, monitors CPU, parses time splits
  monitor.py           # /proc CPU/disk sampling
  models.py            # Pydantic models (TimeSplit, SweepData, etc.)
  inference_metrics.py # vLLM Prometheus metrics scraping (KV cache, TTFT, etc.)
  enroot_env.py        # Harbor BaseEnvironment plugin using Enroot
  docker_env.py        # Harbor BaseEnvironment plugin using Docker
  opensandbox_env.py   # Harbor BaseEnvironment plugin using remote OpenSandbox
  apptainer_env.py     # Harbor BaseEnvironment plugin using Apptainer (see apptainer/)
  apptainer/           # Apptainer image, lifecycle, transfer, and policy components
```

`enroot_env.py` is a Harbor plugin loaded via:

```text
--environment-import-path tasktrace_capture.enroot_env:EnrootEnvironment
```

The CLI adds this import path automatically.

## Current Limitations

- Run only one Enroot or Apptainer TaskTrace-capture instance per node. Concurrent instances can interfere with each other and distort CPU metrics, so use separate Slurm jobs for parallel sweeps.
- Docker supports multiple instances with `--allow-parallel-docker`.
- OpenSandbox runs remotely and is not subject to the one-instance-per-node restriction.
- `network_mode="no-network"` and legacy `allow_internet=False` are rejected by default because Enroot requires privileges unavailable on typical HPC targets to isolate networking. For compatibility testing only, `--enroot-network-policy best-effort` runs with public host networking, emits a warning, and records the unenforced policy in each trial's `network_policy.json`.
- Enroot does not support arbitrary multi-stage Dockerfile builds; it uses the base image and supported task setup paths.
- Apptainer requires a prepared, native-architecture SIF and direct-SIF host support.
- LLM vs tool time split is available for `terminus-2`, `claude-code`, and `mini-swe-agent`; other Harbor agents may run without this breakdown.
