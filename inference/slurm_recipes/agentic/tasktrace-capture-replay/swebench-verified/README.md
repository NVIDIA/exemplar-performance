# TaskTrace recipe — live capture and replay of a SWE-bench Verified run (Grace CPU)

LLMB recipe that runs one live agentic coding task on an NVIDIA Grace CPU node, captures the trajectory (every tool call the agent issues, with its timing and outcome), and then replays that fresh trajectory's tool side in the same task container. The agent talks to an external OpenAI-compatible endpoint. No GPU usage: the job goes to the CPU partition when `cluster_config.yaml` configures one, else to the GPU partition (see the [replay recipe](../../tasktrace-replay/swebench-verified/README.md#prerequisites)). The recipe checks that the replay reproduces the live run's recorded outcomes and stages the trajectory with a provenance record ready for promotion into `inference/trajectories/`.

## Properties

| Field                   | Value                                                                                                                                                                     |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Workload key**        | `inference_tasktrace_capture_replay`                                                                                                                                      |
| **Capture tool**        | `tasktrace-capture` CLI                                                                                                                                                   |
| **Replay tool**         | `tasktrace` CLI                                                                                                                                                           |
| **Task**                | SWE-bench Verified `django__django-11490` (Harbor dataset `swebench-verified`)                                                                                            |
| **Agent**               | mini-swe-agent, built-in default profile (step limit 0, cost limit 3.0)                                                                                                   |
| **Endpoint**            | public NVIDIA API `https://integrate.api.nvidia.com/v1`, model `openai/nvidia/nemotron-3-ultra-550b-a55b`, key from `$PUBLIC_NVIDIA_API_KEY`, collected by `llmb-install` |
| **Task container**      | `ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11490@sha256:58f9edb89a621c2121dc3132ce311130d61d8a3a8075e250231f2a9d757b9a78`                                |
| **CPU**                 | NVIDIA Grace (Arm Neoverse V2), tested only in GB300 and GB200 nodes                                                                                                      |
| **Runtime**             | enroot (both phases)                                                                                                                                                      |
| **Trajectory selector** | `-s nemotron-3-ultra_miniswe_django-11490` (`<model>_<agent-harness>_<task>`)                                                                                             |

## What it measures

An agent run alternates between waiting on the model (LLM calls) and running tool calls in its sandbox. This recipe runs both halves live once, then re-runs the tool half from the trajectory it just captured:

1. **Capture** (`tasktrace-capture --instrument-shell-commands`): the live run inside the task image, recording per-tool-call wall times, the LLM-versus-tool time split, the verifier's reward, and host telemetry.
2. **Replay** (`tasktrace profile trace`): every captured tool call re-executed in the same image without the LLM wait between calls (pacing `unpaced` by default, set with `RUN_CONF_TASKTRACE_PACING`), checking that each reproduces its recorded outcome.

It answers: does this cluster's sandbox path reproduce a run it just made?

The live run is nondeterministic: its trajectory, and possibly its reward, will differ from the trajectory shipped with the [replay-only recipe](../../tasktrace-replay/swebench-verified/README.md).

### Container environment and isolation

Both phases run each command in the task container the same way, so the replay sees the environment the live run saw:

- Each command starts from the image's `/etc/environment` over a clean base (`HOME` and a default `PATH`), not the job's environment. enroot runs with `ENROOT_MOUNT_HOME=no`, so the host home directory is not mounted.
- The capture records the environment mini-swe-agent's tool commands ran with (for example the `LC_CTYPE=C.UTF-8` its Python sets), minus capture settings, LLM client settings and anything named like a credential, and the replay applies it.
- The capture passes `--source-image-bashrc`, which sets `BASH_ENV=/root/.bashrc` so non-interactive Bash activates the image's `testbed` conda environment. The trajectory header records it, and the replay sets the same value.
- Each command runs in its own PID namespace, so processes it leaves behind end with it. Both tools use `unshare --user --map-current-user` and fall back to `bwrap`; if neither works on the node (unprivileged user namespaces disabled), every container command fails. After each trial the capture kills container processes left running, but only those in its own Slurm job step's cgroup, so other processes in the job, such as the `sybil` service some clusters run in the job's extern step, are never signalled.

## Prerequisites

1. **A Slurm cluster with GB300 or GB200 nodes and enroot.** The recipe uses only their Grace CPUs. The launch script refuses a node whose architecture differs from the arm64 task image.
2. **A public NVIDIA API key** (`nvapi-…`). Generate it at <https://build.nvidia.com/settings/api-keys>.
3. **Compute-node access to the endpoint, `github.com`, `pypi.org`, `astral.sh` and the Ubuntu package mirror.** The agent calls `integrate.api.nvidia.com` by default; the live run fetches the task and its verifier from GitHub and installs the agent harness inside the task container (apt packages, then uv from astral.sh, then the harness from PyPI).

## Prepare environment

Use the **installer** referenced in the [main README](../../../../../README.md) to prepare the recipe environment. The following directory layout and key variables are used:

- `LLMB_INSTALL`: top-level directory for all benchmarking artifacts.
- `LLMB_WORKLOAD`: `${LLMB_INSTALL}/workloads/inference_tasktrace_capture_replay`; the `tasktrace` binary lands under `bin/`, the capture tool's venv under `venv/`, and the seeded image cache under `enroot-cache/`.
- Task image: `${LLMB_INSTALL}/images/epoch-research+swe-bench.eval.arm64.django__django-11490+sha256-58f9edb89a62.sqsh`.

# Run benchmark

## Quickstart

```bash
# Navigate to your installation directory
cd $LLMB_INSTALL

# Submit the benchmark
llmb-run submit -w inference_tasktrace_capture_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1
```

`--scale 1` is the only accepted scale: one node capturing and replaying one trajectory. A run takes 20 to 60 minutes, mostly the live agent run, whose length depends on how fast the endpoint answers. `launch.sh` requests 90 minutes.

`RUN_CONF_TASKTRACE_TIMEOUT` sets the replay's per-action timeout in seconds, which otherwise comes from the trajectory (mini-swe-agent's 30 s). Raising it is not a fix for fidelity failures: an action that timed out in the live run counts as a mismatch if the replay lets it finish.

`RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS=omit` records the trajectory without its measured timings, like the shipped one. The replay then runs unpaced (it refuses `recorded`), and the report shows the replay's own tool time instead of comparing it with the live run's.

### Additional Slurm parameters

`llmb-run submit` has first-class flags for the common Slurm options and `--slurm-arg` for the rest; they go on the `sbatch` command line, so they override the `#SBATCH` directives in `launch.sh`. `ADDITIONAL_SLURM_PARAMS` is not used by this recipe.

```bash
# a reservation, or a specific node
llmb-run submit -w inference_tasktrace_capture_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1 --reservation my_reservation
llmb-run submit -w inference_tasktrace_capture_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1 --nodelist node001

# any other sbatch option, without the leading dashes (repeatable)
llmb-run submit -w inference_tasktrace_capture_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1 --slurm-arg cpus-per-task=16 --slurm-arg time=01:00:00

# a different partition for one run
llmb-run submit -w inference_tasktrace_capture_replay -s nemotron-3-ultra_miniswe_django-11490 --scale 1 --env SBATCH_PARTITION=<partition>
```

## Where outputs land

`llmb-run` pre-creates `$LLMB_EXPERIMENT_DIR` under `${LLMB_WORKLOAD}/experiments/`. The recipe writes:

- `summary.json`: live-run facts, outcome verdict, tool-time comparison, image identity check, secret-scan result.
- `trajectories/<model>/<agent-harness>/<dataset>/<task>/`: the staged trajectory, its `.sha256`, and `provenance.json` (see below).
- `capture/`: `tasktrace-capture`'s output tree: `report.json`, `report.txt`, `scaling_results.json`, `telemetry.raw`, and `runs/<job>/<trial>/` with the Harbor trial (`result.json`, `trial.log`, `agent/`, `verifier/`). The trial's `agent/` directory holds the raw shell event log, the finalized trajectory and, unless timings are omitted, its Perfetto timeline. Agent-side installs land here too, so this tree is much larger than a replay-only experiment.
- `profile/pipeline_report.json`: image identity, runtime, replay outcome.
- `profile/replay/replay_report.json`: per-action status, return codes, live and replay wall times, `outcome_matches_source`.
- `secret_scan.json`: files scanned for the API key's value and for the `nvapi-` key pattern, and any redactions made.
- `slurm-<jobid>.out`: job log.

## Viewing results

```bash
python3 $LLMB_REPO/inference/slurm_recipes/agentic/common/tasktrace_recipe.py verdict "$(llmb-run jobs log <job_id> --dir)"
```

The report, the per-run table for a whole `experiments/` tree, and what each row and cause means are described once in [Viewing results](../../README.md#viewing-results) in the agentic README.

## Troubleshooting

- Partition, GPU-request and NVLink install problems: see Troubleshooting in the [agentic README](../../README.md#troubleshooting).
- `tasktrace-capture missing` or `tasktrace binary missing`: the matching install task did not complete. Check `${LLMB_WORKLOAD}/uv/uv-install.log`, `${LLMB_WORKLOAD}/rust/rustup-install.log`, and re-run `llmb-install`.
- `$PUBLIC_NVIDIA_API_KEY is empty`: the key is missing from the `environment` section of `cluster_config.yaml`. Re-run `llmb-install` (it prompts for it), add it there by hand, pass `--env PUBLIC_NVIDIA_API_KEY=…` to `llmb-run submit`, or point `RUN_CONF_API_KEY_VAR` at another variable.
- `endpoint check failed with HTTP 401`: the key was rejected. For build.nvidia.com, regenerate it while signed in under the NV-Developer org.
- `endpoint check failed with HTTP 404`: the model id is not served at the base URL. `curl https://integrate.api.nvidia.com/v1/models` lists the public catalog; the value after the `openai/` prefix must match one of its ids.
- `endpoint check: WARNING HTTP 429`: the key works but the endpoint is rate limited; the run continues. Ask in `#api-catalog` for a rate-limit exemption on the model with your NGC org's NCA ID if the agent keeps hitting it.
- `RUN_CONF_MODEL must be <provider>/<model id>`: mini-swe-agent needs a LiteLLM provider prefix; use `openai/<catalog id>`.
- `capture produced no finalized trajectory`: the live run did not reach a finalized trajectory. The job log lists the trial directories and the tail of each `trial.log`; common causes are endpoint authentication or model-id errors, and missing egress to `github.com` (Harbor task fetch) or `pypi.org` (agent install inside the container).
- `No other tasktrace-capture process on this node` failing in preflight: a previous capture is still running on the node; the tool allows one instance per node.
- `container command completed without a replay report` in `profile/`: the controller binary could not run inside the task image. The install builds a static musl binary for the compute nodes' CPU architecture (`LLMB_NODE_ARCHITECTURE`) to prevent this; check `${LLMB_WORKLOAD}/bin/tasktrace.version` for the target and rebuild if the binary was copied from elsewhere.
- `image_identity_check.match` false: the replay saw a different `.sqsh` than the capture. Check `${LLMB_WORKLOAD}/enroot-cache/` for a stale import and remove it so the recipe re-seeds the link to the installer's image.
- Outcome `FAIL` with `exit_code_mismatch` on network actions: the compute node could not reach the package index the live run used minutes earlier.
