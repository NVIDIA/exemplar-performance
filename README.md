# Exemplar Performance

Performance Recipes are ready-to-use templates for evaluating the performance of specific AI use cases across hardware and software combinations. These containerized recipes allow users to quickly set up and run a standardized benchmarking methodology in their own environment, designed to produce consistent and comparable results across platforms.

These Performance Recipes support performance characterization:

- across a variety of defined AI workloads, including pre-training, fine-tuning, and inference.
- across GPU-based infrastructure, whether running on-premises or with cloud service providers (CSPs).

Each recipe maps to one workload and can be run at various cluster scales and precisions. These workloads are tested against NVIDIA Reference Architectures to establish baselines for comparison. The resulting performance metrics are collected from production environments and are subject to real-world variability.

For validation guidance, see the [Exemplar Training Guide](Exemplar_training_validation.md) and the [Exemplar Inference Guide](Exemplar_inference_validation.md), which lists available recipes by chip and scheduler.

## Prerequisites

To use the Performance Recipes, make sure the following prerequisites are available on your cluster:

### General Prerequisites

- Bash 4.2 or newer
- [Git LFS](https://git-lfs.com/)
- [NGC Registry Access](https://org.ngc.nvidia.com/setup)
- Python 3.12.x
- [CUDA](https://developer.nvidia.com/cuda-downloads): at least 12.3, recommended 12.8 or newer
- [NV Driver](https://www.nvidia.com/en-us/drivers/): at least 535.129.03, recommended 570.172.08 or newer
- [OFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/): 5.9-0.5.6.0.127 or newer
- [NCCL](https://developer.nvidia.com/nccl/nccl-download): 2.19.4 or newer

### Cluster-Specific Prerequisites

Depending on your cluster's job scheduler, make sure the following requirements are met:

- **Slurm Clusters**

  - Version 22.x or newer
  - `task/affinity` plugin required for process pinning
  - PMIx support required; Slurm must be built with `--with-pmix` (verify with `srun --mpi=list`)
  - [Enroot](https://github.com/NVIDIA/enroot/) 4.0.0 or newer
    - Enroot [extra hooks](https://github.com/NVIDIA/enroot/tree/main/conf/hooks/extra) (for example, `50-slurm-pytorch.sh`) must be installed under `/etc/enroot/hooks.d/`; these hooks are required for PyTorch distributed bootstrap.
  - [Pyxis](https://github.com/NVIDIA/pyxis)

- **Kubernetes Clusters (nvcre)**

  - `kubectl` on PATH with a valid kubeconfig
  - `nvcrectl` on PATH
  - A gang scheduler for multi-node jobs — [KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler) is the default and must be installed on the cluster (see [Install the KAI Scheduler](#install-the-kai-scheduler))
  - A Hugging Face token — pods download gated model files at runtime

### Storage Requirements

For Exemplar or pretraining-only benchmarking, plan for at least **300 GB** of storage. This covers the typical space needed for recipe setup, including container images and Python environments.

Allocate more storage if you enable profiling, keep many profiled runs, or install non-pretraining workloads that require datasets or checkpoints. Profiling output can range from several GB to hundreds of GB per run, depending on the workload.

## Quick Start Guide

> **Important:** Before proceeding with installation, please review the [Known Issues](#known-issues) section.

1. Clone the repository:

   ```bash
   git clone https://github.com/NVIDIA/exemplar-performance.git
   cd exemplar-performance
   ```

2. Set up Hugging Face access (required):
   Most recipes fetch model metadata, such as tokenizers and configs, from the Hugging Face Hub during installation. Unauthenticated access is heavily rate-limited and commonly causes installation failures.

   - Create a Hugging Face account if you do not have one.
   - Create an access token in [Hugging Face settings](https://huggingface.co/settings/tokens).
   - Keep the Hugging Face token handy. The installer will prompt you for `HF_TOKEN`. If `HF_TOKEN` is already set in your environment, the installer will use it as the default.

   **Gated model access (important):** Some recipes use gated Hugging Face model repositories, such as Llama. Even with `HF_TOKEN`, you must request repo access separately. **Approvals are not immediate**; request access early.

   See [Model Access Requirements](#model-access-requirements) for the list of recipes that require additional approval.

3. Check cluster runtime configuration:

   If you are installing on a Slurm cluster, confirm the required runtime settings before running the installer. Incorrect defaults can cause install and setup jobs to fail.

   <details>

   <summary>Enroot / Pyxis settings</summary>

   **enroot.conf**

   Set these values in `/etc/enroot/enroot.conf`:

   - `ENROOT_ROOTFS_WRITABLE yes`
   - `ENROOT_REMAP_ROOT yes`

   **environ.d**

   Set cluster-specific environment variables needed inside containers in `/etc/enroot/environ.d/*.env`.
   A common issue is a missing `NCCL_IB_HCA`, which can cause multi-node NCCL jobs to fail or pick the wrong HCAs.

   **Home mounts**

   All recipes use `--no-container-mount-home` to prevent the host environment from overriding the container environment.

   </details>

4. Run the installer:

   **Important:** Installation may take several hours, depending on selected recipes, internet speed, and current node resources. Consider using a tool like `tmux` or `screen`.

   The installer ensures a compatible `uv` version is available (installing a specific version if the release requires one), sets up a supported Python environment, and launches the interactive installer. If your active environment is compatible, the installer reuses it; otherwise, it creates a uv-managed `../llmb_venv` one directory above the repo.

   ```bash
   ./install.sh
   ```

   To reuse container images across multiple installs on the same system, pass a writable shared image folder (in case of Slurm):

   ```bash
   ./install.sh -i /shared/llmb-images
   ```

   This forwards `-i` to `llmb-install` and avoids downloading images that already exist in that folder.

   The installer will:

   - Install `uv` (the required package manager) if it is not already present
   - Set up a Python 3.12.x virtual environment (reusing your current one if compatible)
   - Install the CLI tools (`llmb-run`, `llmb-install`)
   - Prompt you to configure your cluster and select workloads to install

   > **Note:** For detailed installation options, workload-specific virtual environments, and troubleshooting, see the [Installer README](cli/llmb-install/README.md).

5. Validate your cluster configuration:

   Before running your first benchmark, we recommend running the system info recipe to collect basic system information and check for common cluster configuration issues (applicable for Slurm clusters only):

   ```bash
   cd $LLMB_INSTALL
   llmb-run submit -w microbenchmark_system_info --scale <num_gpus_per_node>
   ```

   This recipe collects host and container diagnostics, including `lscpu`, NUMA information, `enroot.conf`, `environ.d`, and a basic container startup check.

6. Try the system collector (recommended):

   The collector is opt-in for this initial release. If you installed the `microbenchmark_llmb_collector` workload, run it during initial testing to capture a system report:

   ```bash
   cd "$LLMB_INSTALL"
   llmb-run submit -w microbenchmark_llmb_collector --scale 1
   ```

   The report is saved as `llmb-collector.json` in the experiment directory. See the [collector recipe](microbenchmarks/llmb_collector/README.md) for details.

7. Run a benchmark:

   ```bash
   # Navigate to your installed workload directory
   cd $LLMB_INSTALL

   # Example: Run Llama 3.1 405B pretraining on 256 GPUs with FP8 precision
   llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256
   ```

   To run one workload/model-size target, use `-w <workload> -s <model-size>`. For target lists, omit `-s` and pass comma-separated `-w` entries: `pretrain_llama3.1_70b,pretrain_deepseek-v3` selects one Llama 3.1 model size plus DeepSeek V3, while `pretrain_llama3.1,pretrain_deepseek-v3` includes all installed Llama 3.1 model sizes plus DeepSeek V3.

8. (Optional) (For Slurm clusters only) Package results for sharing:

   When you're ready to share results — for example, as part of [Exemplar Cloud certification](Exemplar_training_validation.md) — bundle all experiment data into a single archive:

   ```bash
   llmb-run archive
   ```

   See the [llmb-run README](cli/llmb-run/README.md#archive-command) for details and options.

### Kubernetes (nvcre) Installation

Before running `./install.sh`, make sure `kubectl` and `nvcrectl` are on your `PATH`, your kubeconfig points at the target cluster, KAI Scheduler is installed on the cluster, and you have a Hugging Face token (see [Cluster-Specific Prerequisites](#cluster-specific-prerequisites)).

#### Install the KAI Scheduler

Multi-node WorkloadRuns need gang scheduling so that all pods of a job start together. The installer records `gang_scheduler_name: kai-scheduler` in `cluster_config.yaml` and each WorkloadRun requests it, but the installer does not install or check for [KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler). If it is missing, job pods are never scheduled and stay `Pending`.

Check whether it is already installed:

```bash
kubectl get pods -n kai-scheduler
```

If the namespace is empty or missing, install it with Helm (cluster-admin permissions required):

```bash
export KAI_VERSION=v0.18.2

helm upgrade -i kai-scheduler \
  oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  -n kai-scheduler --create-namespace \
  --version $KAI_VERSION
```

Wait for the scheduler pods to become `Running` before continuing. If your cluster uses a different gang scheduler, change `gang_scheduler_name` in `cluster_config.yaml` to its scheduler name after installation.

#### Run the Installer

Choose **Kubernetes** as the execution target when the installer asks. It prompts for:

| Prompt                     | What to enter                                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Namespace                  | Namespace that WorkloadRuns are submitted into (default: `default`).                                                      |
| Kubeconfig path            | Leave blank to use `$KUBECONFIG` or `~/.kube/config`.                                                                     |
| Image pull secret name(s)  | Comma-separated secrets used to pull the container image. Leave blank if none.                                            |
| NGC API key                | Asked only if you named pull secrets. The installer uses it to create the secrets in the namespace and does not store it. |
| PVC name                   | Optional PersistentVolumeClaim used to sync job code and configs and to keep profiling output. Leave blank if none.       |
| Run `nvcrectl setup init`? | Answer **No** if the nvcre operator is already installed. Answer **Yes** to install or upgrade it.                        |

Settings are saved in `cluster_config.yaml` under `kubernetes:`. Workloads that support Kubernetes are marked in the [Available Benchmarks](#available-benchmarks) tables.

### Checking Job Progress and Results on Kubernetes

Submit as usual from `$LLMB_INSTALL`:

```bash
llmb-run submit -w pretrain_qwen3 -s 30b_a3b --dtype bf16 --scale 8
```

List jobs, with status, elapsed time, and results (`s/iter`, `TFLOPS/GPU`):

```bash
llmb-run jobs
```

Show one job, or follow its log while it runs:

```bash
llmb-run jobs show <job_id>
llmb-run jobs log <job_id> --follow
```

To inspect the pods directly:

```bash
kubectl get pods -n <namespace>
kubectl logs -f <pod-name> -n <namespace>
```

When a job reaches a terminal state, `llmb-run` pulls the logs of all its pods into the experiment directory under `pod_logs/`, then cancels the WorkloadRun to release cluster resources. The results columns are filled from the pulled rank-0 log. If the log pull fails, the WorkloadRun is kept so you can retrieve the logs manually.

See the [llmb-run jobs command reference](cli/llmb-run/README.md#jobs-command) for all options.

### Shell Completion (Optional)

Enable tab completion for `llmb-run` commands and options:

```bash
llmb-run --install-completion
```

Restart your shell after installation for changes to take effect.

### Directory Layout and Key Variables

After running the installer, the following directory structure is created:

- `LLMB_REPO`: Directory containing the clone of the recipe repository.
- `LLMB_INSTALL`: Top-level directory for all benchmarking artifacts, including images, datasets, venvs, and workloads.
- `LLMB_WORKLOAD`: Workload-specific directory, for example `${LLMB_INSTALL}/workloads/pretrain_llama3.1`.
- Results, logs, and checkpoints are stored under subfolders of `LLMB_WORKLOAD` (see below).

**Example structure:**

```
$LLMB_INSTALL/
  ├── bin/
  ├── images/
  ├── datasets/
  ├── venvs/
  └── workloads/
        └── pretrain_llama3.1/   # <- $LLMB_WORKLOAD
              ├── NeMo/
              ├── ...
              └── experiments/
```

`LLMB_REPO`, `LLMB_INSTALL`, and `LLMB_WORKLOAD` are shorthand terms used throughout this README. The installer asks for an installation directory and the launcher passes that path to workload scripts as `LLMB_INSTALL`.

## Workload Resources Overview

Each recipe includes documentation and supporting scripts for its workload:

- **Configuration details**: Software and hardware setup information.
- **Performance scripts**: Scripts to generate and analyze performance results.

The recipe documentation describes supported configurations and performance metrics for the workload, focusing on speed measurements such as TFLOPS per GPU, time per training step, and tokens processed per second.

## Available Benchmarks

The following tables list the available benchmarks and their configurations.

**Note:** The "Scale (# of GPUs)" column indicates the minimum supported scale and the maximum scale tested for each workload. The recipes may function at larger scales unless otherwise noted in the workload-specific README, although they have not been explicitly validated beyond the listed maximum.

### GB300 Workloads

|      Type      |    Framework    |                                                    Model                                                     |  Container Version  | Model Size | Scale (# of GPUs) |    Precision     | Model Access Required | Checkpointing |   Cluster Type    |
| :------------: | :-------------: | :----------------------------------------------------------------------------------------------------------: | :-----------------: | :--------: | :---------------: | :--------------: | :-------------------: | :-----------: | :---------------: |
|    Pretrain    | Megatron-Bridge |                          [GPT OSS 20B (MLPerf)](gpt-oss/pretrain/mlperf/README.md)                           |      26.08.01       |    20B     |       8-512       |    NVFP4, FP8    |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                          [GPT OSS 120B](gpt-oss/pretrain/megatron_bridge/README.md)                          |      26.08.01       |    120B    |      64-576       |    FP8, BF16     |          No           |      No       | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |                        [DeepSeek V3](deepseek_v3/pretrain/megatron_bridge/README.md)                         |      26.08.01       |    671B    |      128-512      | NVFP4, FP8, BF16 |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                          [DeepSeek V4 Flash](deepseek_v4/pretrain/flash/README.md)                           |      26.08.01       |    284B    |      128-512      |       FP8        |          Yes          |      No       | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |                           [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)                           |      26.08.01       |    405B    |      256-512      |    NVFP4, FP8    |          Yes          |      Yes      |       Slurm       |
|    Pretrain    | Megatron-Bridge |                           [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)                           |      26.08.01       |    70B     |      64-512       |    NVFP4, FP8    |          Yes          |      Yes      | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |                           [Llama 3.1 (MLPerf)](llama3.1/pretrain/mlperf/README.md)                           |      26.08.01       |     8B     |       8-72        |      NVFP4       |          Yes          |      Yes      | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |                                      [Qwen3](qwen3/pretrain/README.md)                                       |      26.08.01       |    235B    |      256-512      | NVFP4, FP8, BF16 |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                                      [Qwen3](qwen3/pretrain/README.md)                                       |      26.08.01       |    30B     |       8-64        |    FP8, BF16     |          Yes          |      No       | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |                               [Nemotron 3 Super](nemotron3/pretrain/README.md)                               |      26.08.01       |    120B    |      64-512       |    FP8, BF16     |          No           |      No       | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |                               [Nemotron 3 Ultra](nemotron3/pretrain/README.md)                               |      26.08.01       |    550B    |        256        |       FP8        |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                               [Nemotron 3.5 Lightning](nemotron3.5/README.md)                                |      26.08.01       |    30B     |       8-64        |    FP8, BF16     |          No           |      No       | Slurm, Kubernetes |
|    Finetune    | Megatron-Bridge |                                     [Llama 3](llama3/finetune/README.md)                                     |      26.08.01       |    70B     |       8-16        |    FP8, BF16     |          Yes          |      No       |       Slurm       |
|    Finetune    | Megatron-Bridge |                                     [Llama 2](llama2/finetune/README.md)                                     |      26.08.01       |    70B     |       4-512       |       FP8        |          Yes          |      No       |       Slurm       |
|   Inference    | TRT-LLM Dynamo  |            [GLM-5](inference/slurm_recipes/llm-serving/glm5/disagg/gb300/trtllm_dynamo/README.md)            |     1.1.0-dev.3     |    744B    |       32-40       |      NVFP4       |          Yes          |      No       |       Slurm       |
|   Inference    |  SGLang Dynamo  |            [GLM-5](inference/slurm_recipes/llm-serving/glm5/disagg/gb300/sglang_dynamo/README.md)            |        1.3.0        |    744B    |       40-72       |       FP8        |          Yes          |      No       |       Slurm       |
|   Inference    |  SGLang Dynamo  | [Nemotron 3 Ultra](inference/slurm_recipes/llm-serving/nemotron3_ultra/disagg/gb300/sglang_dynamo/README.md) | 1.4.0-inkling-dev.1 |    550B    |       8-48        |      NVFP4       |          No           |      No       |       Slurm       |
|   Inference    |  SGLang Dynamo  |                   [DeepSeek V4 Flash](inference/slurm_recipes/llm-serving/dsv4/README.md)                    |        1.5.1        |    304B    |         8         |      MXFP4       |          Yes          |      No       |       Slurm       |
|   Inference    |    TaskTrace    |                [Nemotron 3 Ultra SWE-bench replay](inference/slurm_recipes/agentic/README.md)                |       in-tree       |    550B    |         0         |      NVFP4       |          No           |      No       |       Slurm       |
|   Inference    |    TaskTrace    |               [Nemotron 3 Ultra SWE-bench capture](inference/slurm_recipes/agentic/README.md)                |       in-tree       |    550B    |         0         |      NVFP4       |          No           |      No       |       Slurm       |
|   Inference    |    TaskTrace    |                   [Nemotron 3 Ultra GPU replay](inference/slurm_recipes/agentic/README.md)                   |     vLLM 0.20.1     |    550B    |         4         |      NVFP4       |          No           |      No       |       Slurm       |
|   Inference    |    TaskTrace    |                  [Nemotron 3 Ultra GPU capture](inference/slurm_recipes/agentic/README.md)                   |     vLLM 0.20.1     |    550B    |         4         |      NVFP4       |          No           |      No       |       Slurm       |
| Microbenchmark |     TRT-LLM     |                              [GPT-OSS](microbenchmarks/cpu_overhead/README.md)                               |      1.1.0rc5       |    120B    |        1-4        |      MXFP4       |          Yes          |      No       |       Slurm       |

### GB200 Workloads

|      Type      |    Framework    |                                             Model                                             |    Container Version    | Model Size | Scale (# of GPUs) |    Precision     | Model Access Required | Checkpointing |   Cluster Type    |
| :------------: | :-------------: | :-------------------------------------------------------------------------------------------: | :---------------------: | :--------: | :---------------: | :--------------: | :-------------------: | :-----------: | :---------------: |
|    Pretrain    | Megatron-Bridge |                   [GPT OSS 20B (MLPerf)](gpt-oss/pretrain/mlperf/README.md)                   |        26.08.01         |    20B     |       8-512       |    NVFP4, FP8    |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                  [GPT OSS 120B](gpt-oss/pretrain/megatron_bridge/README.md)                   |        26.08.01         |    120B    |      64-512       |    FP8, BF16     |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                   [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)                    |        26.08.01         |    405B    |      256-512      |    NVFP4, FP8    |          Yes          |      Yes      |       Slurm       |
|    Pretrain    | Megatron-Bridge |                   [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)                    |        26.08.01         |    70B     |      64-512       |    NVFP4, FP8    |          Yes          |      Yes      |       Slurm       |
|    Pretrain    | Megatron-Bridge |                   [Llama 3.1 (MLPerf)](llama3.1/pretrain/mlperf/README.md)                    |        26.08.01         |     8B     |       8-72        |      NVFP4       |          Yes          |      Yes      |       Slurm       |
|    Pretrain    | Megatron-Bridge |                               [Qwen3](qwen3/pretrain/README.md)                               |        26.08.01         |    235B    |      256-512      | NVFP4, FP8, BF16 |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                               [Qwen3](qwen3/pretrain/README.md)                               |        26.08.01         |    30B     |       8-64        |    FP8, BF16     |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                 [DeepSeek V3](deepseek_v3/pretrain/megatron_bridge/README.md)                 |        26.08.01         |    671B    |      128-512      | NVFP4, FP8, BF16 |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                   [DeepSeek V4 Flash](deepseek_v4/pretrain/flash/README.md)                   |        26.08.01         |    284B    |      128-512      |       FP8        |          Yes          |      No       | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |                       [Nemotron 3 Super](nemotron3/pretrain/README.md)                        |        26.08.01         |    120B    |      64-512       |    FP8, BF16     |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                       [Nemotron 3 Ultra](nemotron3/pretrain/README.md)                        |        26.08.01         |    550B    |        256        |       FP8        |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |                        [Nemotron 3.5 Lightning](nemotron3.5/README.md)                        |        26.08.01         |    30B     |       8-64        |    FP8, BF16     |          No           |      No       | Slurm, Kubernetes |
|    Finetune    | Megatron-Bridge |                             [Llama 3](llama3/finetune/README.md)                              |        26.08.01         |    70B     |       8-16        |    FP8, BF16     |          Yes          |      No       |       Slurm       |
|    Finetune    | Megatron-Bridge |                             [Llama 2](llama2/finetune/README.md)                              |        26.08.01         |    70B     |       4-512       |       FP8        |          Yes          |      No       |       Slurm       |
|   Inference    | TRT-LLM Dynamo  | [Kimi-K2.6](inference/slurm_recipes/llm-serving/kimi2.6/disagg/gb200/trtllm_dynamo/README.md) |       1.1.0-dev.2       |     1T     |       24-48       |      NVFP4       |          Yes          |      No       |       Slurm       |
|   Inference    |  SGLang Dynamo  |            [DeepSeek V4 Flash](inference/slurm_recipes/llm-serving/dsv4/README.md)            |          1.5.1          |    304B    |         8         |      MXFP4       |          Yes          |      No       |       Slurm       |
|   Inference    |     SGLang      |             [Qwen3 long context](inference/slurm_recipes/long-context/README.md)              |         v0.5.11         |    32B     |         4         |       BF16       |          Yes          |      No       |       Slurm       |
|   Inference    |    TaskTrace    |        [Nemotron 3 Ultra SWE-bench replay](inference/slurm_recipes/agentic/README.md)         |         in-tree         |    550B    |         0         |      NVFP4       |          No           |      No       |       Slurm       |
|   Inference    |    TaskTrace    |        [Nemotron 3 Ultra SWE-bench capture](inference/slurm_recipes/agentic/README.md)        |         in-tree         |    550B    |         0         |      NVFP4       |          No           |      No       |       Slurm       |
|   Inference    |    TaskTrace    |           [Nemotron 3 Ultra GPU replay](inference/slurm_recipes/agentic/README.md)            |       vLLM 0.20.1       |    550B    |         4         |      NVFP4       |          No           |      No       |       Slurm       |
|   Inference    |    TaskTrace    |           [Nemotron 3 Ultra GPU capture](inference/slurm_recipes/agentic/README.md)           |       vLLM 0.20.1       |    550B    |         4         |      NVFP4       |          No           |      No       |       Slurm       |
| Microbenchmark |     TRT-LLM     |                       [GPT-OSS](microbenchmarks/cpu_overhead/README.md)                       |        1.1.0rc5         |    120B    |        1-4        |      MXFP4       |          Yes          |      No       |       Slurm       |
|       RL       |     NeMo-RL     |                            [DeepSeek V3](deepseek_v3/rl/README.md)                            |          0.6.0          |    671B    |        256        |       BF16       |          Yes          |      No       |       Slurm       |
|       RL       |     NeMo-RL     |                          [Nemotron 3 Super](nemotron3/rl/README.md)                           | v0.5.0.nemotron_3_super |    120B    |        72         |       BF16       |          Yes          |      No       |       Slurm       |

### B300 Workloads

|      Type      |    Framework    |                             Model                             | Container Version | Model Size | Scale (# of GPUs) |    Precision     | Model Access Required | Checkpointing |   Cluster Type    |
| :------------: | :-------------: | :-----------------------------------------------------------: | :---------------: | :--------: | :---------------: | :--------------: | :-------------------: | :-----------: | :---------------: |
|    Pretrain    | Megatron-Bridge |  [GPT OSS 120B](gpt-oss/pretrain/megatron_bridge/README.md)   |     26.08.01      |    120B    |      64-512       |       BF16       |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |   [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)    |     26.08.01      |    405B    |      256-512      |    NVFP4, FP8    |          Yes          |      Yes      |       Slurm       |
|    Pretrain    | Megatron-Bridge |   [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)    |     26.08.01      |    70B     |      64-512       |    NVFP4, FP8    |          Yes          |      Yes      |       Slurm       |
|    Pretrain    | Megatron-Bridge |               [Qwen3](qwen3/pretrain/README.md)               |     26.08.01      |    235B    |      256-512      | NVFP4, FP8, BF16 |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |               [Qwen3](qwen3/pretrain/README.md)               |     26.08.01      |    30B     |       8-64        |    FP8, BF16     |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge | [DeepSeek V3](deepseek_v3/pretrain/megatron_bridge/README.md) |     26.08.01      |    671B    |      128-512      | NVFP4, FP8, BF16 |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |   [DeepSeek V4 Flash](deepseek_v4/pretrain/flash/README.md)   |     26.08.01      |    284B    |      128-512      |       FP8        |          Yes          |      No       | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |       [Nemotron 3 Super](nemotron3/pretrain/README.md)        |     26.08.01      |    120B    |      64-512       |       BF16       |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |        [Nemotron 3.5 Lightning](nemotron3.5/README.md)        |     26.08.01      |    30B     |       8-64        |    FP8, BF16     |          No           |      No       | Slurm, Kubernetes |
|    Finetune    | Megatron-Bridge |             [Llama 3](llama3/finetune/README.md)              |     26.08.01      |    70B     |       8-16        |    FP8, BF16     |          Yes          |      No       |       Slurm       |
| Microbenchmark |     TRT-LLM     |       [GPT-OSS](microbenchmarks/cpu_overhead/README.md)       |     1.1.0rc5      |    120B    |        1-4        |      MXFP4       |          Yes          |      No       |       Slurm       |

### B200 Workloads

|      Type      |    Framework    |                             Model                             |    Container Version    | Model Size | Scale (# of GPUs) |    Precision     | Model Access Required | Checkpointing |   Cluster Type    |
| :------------: | :-------------: | :-----------------------------------------------------------: | :---------------------: | :--------: | :---------------: | :--------------: | :-------------------: | :-----------: | :---------------: |
|    Pretrain    | Megatron-Bridge |  [GPT OSS 120B](gpt-oss/pretrain/megatron_bridge/README.md)   |        26.08.01         |    120B    |      64-512       |       BF16       |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |   [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)    |        26.08.01         |    405B    |      256-512      |    NVFP4, FP8    |          Yes          |      Yes      |       Slurm       |
|    Pretrain    | Megatron-Bridge |   [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)    |        26.08.01         |    70B     |      64-512       |    NVFP4, FP8    |          Yes          |      Yes      |       Slurm       |
|    Pretrain    | Megatron-Bridge |               [Qwen3](qwen3/pretrain/README.md)               |        26.08.01         |    235B    |      256-512      | NVFP4, FP8, BF16 |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |               [Qwen3](qwen3/pretrain/README.md)               |        26.08.01         |    30B     |       8-64        |    FP8, BF16     |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge | [DeepSeek V3](deepseek_v3/pretrain/megatron_bridge/README.md) |        26.08.01         |    671B    |      128-512      | NVFP4, FP8, BF16 |          Yes          |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |   [DeepSeek V4 Flash](deepseek_v4/pretrain/flash/README.md)   |        26.08.01         |    284B    |      128-512      |       FP8        |          Yes          |      No       | Slurm, Kubernetes |
|    Pretrain    | Megatron-Bridge |       [Nemotron 3 Super](nemotron3/pretrain/README.md)        |        26.08.01         |    120B    |      64-512       |       BF16       |          No           |      No       |       Slurm       |
|    Pretrain    | Megatron-Bridge |        [Nemotron 3.5 Lightning](nemotron3.5/README.md)        |        26.08.01         |    30B     |       8-64        |    FP8, BF16     |          No           |      No       | Slurm, Kubernetes |
|    Finetune    | Megatron-Bridge |             [Llama 3](llama3/finetune/README.md)              |        26.08.01         |    70B     |       8-16        |    FP8, BF16     |          Yes          |      No       |       Slurm       |
| Microbenchmark |     TRT-LLM     |       [GPT-OSS](microbenchmarks/cpu_overhead/README.md)       |        1.1.0rc5         |    120B    |        1-4        |      MXFP4       |          Yes          |      No       |       Slurm       |
|       RL       |     NeMo-RL     |          [Nemotron 3 Super](nemotron3/rl/README.md)           | v0.5.0.nemotron_3_super |    120B    |        72         |       BF16       |          Yes          |      No       |       Slurm       |

### H100 Workloads

Baseline performance metrics were collected using workloads on the NVIDIA DGX H100 Reference Architecture. For more information, see [DGX H100 Systems](https://blogs.nvidia.com/blog/dgx-h100-systems-shipping/).

|      Type      |    Framework    |                             Model                             | Container Version | Model Size | Scale (# of GPUs) | Precision | Model Access Required | Checkpointing | Cluster Type |
| :------------: | :-------------: | :-----------------------------------------------------------: | :---------------: | :--------: | :---------------: | :-------: | :-------------------: | :-----------: | :----------: |
|    Pretrain    | Megatron-Bridge |  [GPT OSS 120B](gpt-oss/pretrain/megatron_bridge/README.md)   |     26.06.01      |    120B    |      64-512       |   BF16    |          No           |      No       |    Slurm     |
|    Pretrain    | Megatron-Bridge |   [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)    |     26.06.01      |    405B    |       1024        | FP8, BF16 |          Yes          |      No       |    Slurm     |
|    Pretrain    | Megatron-Bridge |   [Llama 3.1](llama3.1/pretrain/megatron_bridge/README.md)    |     26.06.01      |    70B     |      64-512       | FP8, BF16 |          Yes          |      No       |    Slurm     |
|    Pretrain    | Megatron-Bridge |               [Qwen3](qwen3/pretrain/README.md)               |     26.06.01      |    235B    |      256-512      |   BF16    |          Yes          |      No       |    Slurm     |
|    Pretrain    | Megatron-Bridge |               [Qwen3](qwen3/pretrain/README.md)               |     26.06.01      |    30B     |       16-64       |   BF16    |          Yes          |      No       |    Slurm     |
|    Pretrain    | Megatron-Bridge | [DeepSeek V3](deepseek_v3/pretrain/megatron_bridge/README.md) |     26.06.01      |    671B    |       1024        | FP8, BF16 |          Yes          |      No       |    Slurm     |
|    Pretrain    | Megatron-Bridge |        [Nemotron 3.5 Lightning](nemotron3.5/README.md)        |     26.08.01      |    30B     |       16-64       |   BF16    |          No           |      No       |    Slurm     |
|    Finetune    | Megatron-Bridge |             [Llama 3](llama3/finetune/README.md)              |     26.06.01      |    70B     |       8-16        | FP8, BF16 |          Yes          |      No       |    Slurm     |
| Microbenchmark |     TRT-LLM     |       [GPT-OSS](microbenchmarks/cpu_overhead/README.md)       |     1.1.0rc5      |    120B    |        1-4        |   MXFP4   |          Yes          |      No       |    Slurm     |

### Deprecated

These recipes have been removed from the repository. To use one, check out the release tag listed in its **Last Version** column (for example `git checkout v25.08`).

|          Type           |                  Framework                   |       Model       |            Container Version            | Model Size | Scale (# of GPUs) |    Precision     | Model Access Required | Checkpointing |   Cluster Type    | Last Version |
| :---------------------: | :------------------------------------------: | :---------------: | :-------------------------------------: | :--------: | :---------------: | :--------------: | :-------------------: | :-----------: | :---------------: | :----------: |
|       Finetuning        |                      HF                      |      Llama 2      |                24.02-py3                |    70B     |       8-512       |    FP8, BF16     |          Yes          |      No       |       Slurm       |   25.01.1    |
|       Finetuning        |                      HF                      |      Mistral      |                24.02-py3                |     7B     |       8-256       |    FP8, BF16     |          Yes          |      No       |       Slurm       |   25.01.1    |
|        Pretrain         |                     Jax                      |      Llama 2      |         jax:maxtext-2024-12-09          |    70B     |     128-2048      |    FP8, BF16     |          No           |      No       |       Slurm       |   25.01.1    |
|        Pretrain         |                     Jax                      |       GPT3        |           jax:pax-2024-03-04            |    175B    |     128-2048      |    FP8, BF16     |          No           |      No       |       Slurm       |   25.01.1    |
|        Pretrain         |                   Maxtext                    |      Llama3       |                  25.01                  |    70B     |     128-2048      |    FP8, BF16     |          No           |      No       |       Slurm       |   25.04.02   |
|        Pretrain         |                     NeMo                     |       GPT3        |                  24.12                  |    175B    |     128-2048      |    FP8, BF16     |          No           |      No       |       Slurm       |   25.04.02   |
|        Pretrain         |                     NeMo                     |  Llama4 Maverick  |                25.07.01                 |    400B    |     512-2048      |    FP8, BF16     |          Yes          |      No       |       Slurm       |    25.08     |
| Fine-Tuning (SFT, LoRA) |                     NeMo                     |      Llama 3      |                  24.12                  |  8B, 70B   |       8-32        |    FP8, BF16     |          Yes          |      No       |       Slurm       |   25.04.02   |
|        Finetune         |                NeMo Maverick                 |      Llama4       |                25.07.01                 |    400B    |        256        |    FP8, BF16     |          Yes          |      No       |       Slurm       |    25.08     |
|        Inference        |                     NIM                      |      Llama 3      |                  1.0.3                  |    70B     |         4         |       FP8        |          Yes          |      No       |       Slurm       |   25.05.04   |
|        Inference        |                 NIM, SGLang                  |    DeepSeek R1    |                  1.7.2                  |    671B    |        16         |       FP8        |          No           |      No       |       Slurm       |    25.08     |
|        Inference        | NIM & NeMo Retriever (NVIDIA Enterprise RAG) | Llama 3.1 and 3.2 | instruct:1.3.3, rerank:1.3, embed:1.3.1 |  70b, 1b   |        1-8        |       N/A        |          Yes          |      No       |       Slurm       |    25.08     |
|        Inference        |                   TRT-LLM                    |      Llama 4      |                1.0.0rc1                 |    17b     |         8         |       FP8        |          Yes          |      No       |       Slurm       |    25.08     |
|        Inference        |                   TRT-LLM                    |    DeepSeek R1    |                1.1.0rc5                 |    671B    |       4-16        |    NVFP4, FP8    |          No           |      No       |       Slurm       |   26.05.00   |
|        Inference        |                    Dynamo                    |    DeepSeek R1    |                  0.6.1                  |    671B    |       32-48       |    NVFP4, FP8    |          No           |      No       |       Slurm       |   26.05.00   |
|        Inference        |                    SGLang                    |    DeepSeek R1    |      v0.5.3-cu129, v0.5.3rc0-cu128      |    671B    |        4-8        |      NVFP4       |          No           |      No       |       Slurm       |   26.05.00   |
|        Inference        |                   TRT-LLM                    |     Llama 3.3     |                1.1.0rc5                 |    70B     |        1-4        |    NVFP4, FP8    |          Yes          |      No       |       Slurm       |   26.05.00   |
|        Inference        |               Dynamo + TRT-LLM               |      GPT-OSS      |          0.5.1-rc0.pre3, 0.6.1          |    120B    |        4+         |      MXFP4       |          No           |      No       | Slurm, Kubernetes |   26.05.00   |
|        Pretrain         |                     NeMo                     |   Nemotron4 15B   |                25.09.00                 |    15B     |      16-256       |    FP8, BF16     |          No           |      Yes      |       Slurm       |   26.02.01   |
|        Pretrain         |                     NeMo                     |  Nemotron4 340B   |                25.09.00                 |    340B    |     128-2048      |    FP8, BF16     |          No           |      Yes      |       Slurm       |   26.02.01   |
|        Pretrain         |                     NeMo                     |       Grok1       |                25.09.00                 |    314B    |     128-2048      |    FP8, BF16     |          Yes          |      No       |       Slurm       |   26.02.01   |
|        Pretrain         |               Megatron-Bridge                |     Llama 3.1     |                26.06.01                 |     8B     |       8-128       | NVFP4, FP8, BF16 |          Yes          |      Yes      |       Slurm       |   26.08.01   |
|        Pretrain         |               Megatron-Bridge                |      Kimi-K2      |                26.06.01                 |     1T     |      256-512      |       FP8        |          No           |      No       |       Slurm       |   26.08.01   |
|        Pretrain         |               Megatron-Bridge                |    Nemotron-H     |                26.04.01                 |    56B     |       8-512       |       FP8        |          No           |      No       |       Slurm       |   26.08.01   |
|        Pretrain         |               Megatron-Bridge                |  Nemotron 3 Nano  |                26.06.01                 |    30B     |       8-64        |    FP8, BF16     |          No           |      No       |       Slurm       |   26.08.01   |
|        Pretrain         |                  TorchTitan                  |    DeepSeek V3    |                25.12-py3                |    671B    |      256-512      |    FP8, BF16     |          Yes          |      No       |       Slurm       |   26.08.01   |

## Model Access Requirements

Most recipes require a Hugging Face account and `HF_TOKEN` to fetch model metadata (tokenizer/config) from the Hugging Face Hub without running into strict unauthenticated rate limits.

Some recipes additionally require approval for gated model or dataset repositories. In those cases, the token is necessary but not sufficient — your Hugging Face account must also be granted access to the repository.

**Note:** Approval processes are not immediate and may take some time.

| Recipe Type    | Recipe Name        | HF Token Required | Additional Approval Required | Details/Link for Approval                                                                                                                                                                         |
| :------------- | :----------------- | :---------------- | :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pretrain       | GPT OSS 120B       | Yes               | No                           | [HuggingFace GPT OSS 120B](https://huggingface.co/openai/gpt-oss-120b)                                                                                                                            |
| Pretrain       | Llama 3.1 405B     | Yes               | Yes                          | [HuggingFace Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B)                                                                                                                    |
| Pretrain       | Llama 3.1 8B/70B   | Yes               | Yes                          | [HuggingFace Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B) or [HuggingFace Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B); either grants Llama 3 family access |
| Pretrain       | DeepSeek V3        | Yes               | No                           | N/A                                                                                                                                                                                               |
| Pretrain       | DeepSeek V4 Flash  | Yes               | No                           | [HuggingFace DeepSeek V4 Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)                                                                                                             |
| Pretrain       | Qwen3 235B         | Yes               | No                           | [HuggingFace Qwen3 235B](https://huggingface.co/Qwen/Qwen3-235B-A22B)                                                                                                                             |
| Pretrain       | Qwen3 30B          | Yes               | No                           | [HuggingFace Qwen3 30B](https://huggingface.co/Qwen/Qwen3-30B-A3B)                                                                                                                                |
| Finetune       | Llama 3            | Yes               | Yes                          | [HuggingFace Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)                                                                                                                     |
| Finetune       | Llama 2            | Yes               | Yes                          | [HuggingFace Llama 2 70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)                                                                                                                       |
| Inference      | Qwen3 Long Context | Yes               | Yes                          | [Agentic Coding dataset](https://huggingface.co/datasets/nv-camilom/agentic_coding)                                                                                                               |
| Inference      | GLM-5              | Yes               | No                           | [HuggingFace GLM-5 NVFP4](https://huggingface.co/nvidia/GLM-5-NVFP4)                                                                                                                              |
| Inference      | Kimi-K2.6          | Yes               | No                           | [HuggingFace Kimi-K2.6 NVFP4](https://huggingface.co/nvidia/Kimi-K2.6-NVFP4)                                                                                                                      |
| Inference      | DeepSeek V4 Flash  | Yes               | Yes                          | [HuggingFace DeepSeek V4 Flash 0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) and [Agentic Coding dataset](https://huggingface.co/datasets/nv-camilom/agentic_coding)           |
| Microbenchmark | CPU overhead       | Yes               | No                           | [HuggingFace GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b)                                                                                                                            |

# Reference Infrastructure

Exemplar Performance validates recipes on the reference infrastructure, CSP-specific configurations, and software listed below. These environments provide the baseline configurations used for comparison.

## Peak Theoretical Throughput

The following table shows the peak theoretical throughput (in TFLOPS) for different GPU types and data types. These values represent the maximum computational capacity of each GPU architecture and are used for calculating Model FLOPS Utilization (MFU) in performance analysis.

| Data Type | GB300 | GB200 | B300  | B200 | H100 |
| :-------- | :---: | :---: | :---: | :--: | :--: |
| BF16      | 2450  | 2450  | 2250  | 2250 | 989  |
| FP8       | 4900  | 4900  | 4500  | 4500 | 1979 |
| NVFP4     | 14700 | 9800  | 13500 | 9000 |  -   |

**Note:** These peak theoretical throughput values are based on non-sparse specifications and referenced throughout individual recipe README files for MFU calculations and performance analysis. NVFP4 precision is not supported on Hopper architecture (H100).

## GB300 Reference Architecture

Baseline performance metrics for GB300 workloads were collected using systems equipped with the NVIDIA GB300 Grace Blackwell Superchip. For more information, see [NVIDIA Blackwell Platform](https://www.nvidia.com/en-us/data-center/gb300-nvl72/).

- GB300 Grace Blackwell Superchip
  - CPU: 72 Arm Neoverse V2 cores with 4x 128b SVE2
    - 3.5 GHz (max boost)
    - Low-latency coherent interconnect between Grace CPU and B300 GPUs
    - RAM: 960 GiB LPDDR5X (2x 480 GiB) | 546 GB/s
    - Total Accessible Memory: 2 TiB
    - 64x PCIe Gen5 lanes
  - 2x B300 GPUs
    - 279 GB HBM3e per GPU
    - TDP configurable up to 1,400 W
    - Memory bandwidth 8 TB/s per GPU
- NVLink: NVLink 5th Generation
  - 1.8 TB/s per GPU bandwidth
- System Memory: Coherent memory architecture between Grace CPU and Blackwell GPUs

## GB200 Reference Architecture

Baseline performance metrics for GB200 workloads were collected using the NVIDIA GB200 NVL72 Reference Architecture. For more information, see [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/).

- GB200 Grace Blackwell Superchip
  - CPU: 72 Arm Neoverse V2 cores with 4x 128b SVE2
    - 3.5 GHz (max boost)
    - Low-latency coherent interconnect between Grace CPU and B200 GPUs
    - RAM: 960 GiB LPDDR5X (2x 480 GiB) | 546 GB/s
    - Total Accessible Memory: 1.7 TiB
    - 64x PCIe Gen5 lanes
  - 2x B200 GPUs
    - 186 GB HBM3e per GPU
    - Memory bandwidth 8 TB/s per GPU
- NVLink: NVLink 5th Generation
  - 1.8 TB/s per GPU bandwidth

## B300 Reference Architecture

Baseline performance metrics for B300 workloads were collected using systems equipped with NVIDIA B300 GPUs. For more information, see [NVIDIA DGX B300](https://www.nvidia.com/en-us/data-center/dgx-b300/).

- GPU: 8x B300 270 GB HBM3e (2.1 TB total)
  - TDP: 1100 W
  - Memory bandwidth 7.7 TB/s per GPU
- CPU: Intel Xeon 6776P x2
  - 64 cores per socket
  - 3.9 GHz (max turbo) / 4.6 GHz (priority core turbo, up to 8 cores)
  - RAM: 2 TB DDR5
  - PCIe Gen5
- NVLink: NVLink 5th Generation
  - 1.8 TB/s per GPU bandwidth
- SpectrumX:
  - Compute links: 8x 800 Gbit/s
- System Memory: 2 TB
- Local Storage:
  - 2x 1.9 TB NVMe M.2
  - 8x 3.84 TB NVMe E1.S

## B200 Reference Architecture

Baseline performance metrics for B200 workloads were collected using systems equipped with NVIDIA B200 GPUs. For more information, see [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/).

- GPU: 8x B200 180 GB HBM3e (1.4 TB total)
  - TDP: 1000 W
  - Memory bandwidth 7.7 TB/s per GPU
- CPU: Intel Xeon Platinum 8570 x2
  - 56 cores per socket
  - 4 GHz (max boost)
  - RAM: 1 TiB | 1.6 TB/s per socket
  - 48x PCIe Gen5 lanes
- NVLink: NVLink 5th Generation
  - 1.8 TB/s per GPU bandwidth
  - 18 Links per GPU
- InfiniBand:
  - Compute links: 8x 400 Gbit/s
- System Memory: 2 TB

## H100 Reference Architecture

Baseline performance metrics for H100 workloads were collected using the NVIDIA DGX H100 Reference Architecture. For more information, see [DGX H100 Systems](https://blogs.nvidia.com/blog/dgx-h100-systems-shipping/).

- GPU: 8x H100 80 GB HBM3 (640 GB total)
  - TDP: 700 W
  - Memory bandwidth 3.2 TB/s per GPU
- CPU: 2x Intel Sapphire Rapids, Intel(R) Xeon(R) Platinum 8480C
  - 112 cores (56 cores per CPU)
  - 2.00 GHz (Base), 3.8 GHz (Max boost)
  - NUMA nodes per socket: 1
  - PCIe Gen5
- NVLink: NVLink 4th Generation
  - 900 GB/s per GPU bandwidth
  - 18 Links per GPU
- InfiniBand:
  - Compute links: 8x 400 Gbit/s
  - Storage links: 2x 400 Gbit/s
- System Memory: 2 TB
- Local Storage:
  - 2x 1.92 TB NVMe M.2
  - 8x 3.84 TB NVMe U.2

## CSP Specific Configurations

AI platforms may vary in network fabric, virtualization, and other platform details, and may require different tuning.
For optimal performance, users should use the correct implementation for their platform. The platform-specific tuning below is provided as a starting point. Further tuning may be necessary if the instance type varies from the relevant Reference Architecture.

### AWS

The NeMo images used by the Performance Recipes already include Elastic Fabric Adapter (EFA) support, so no additional setup is required. If NCCL falls back to the socket transport or multi-node performance is unexpectedly low, see [AWS EFA container guidance](docs/csp/aws-efa.md) for how to verify that EFA is active and, if needed, rebuild the image with an updated EFA stack.

### GCP

Ensure that all required preconditions for [GCP cluster deployment](https://cloud.google.com/ai-hypercomputer/docs/create/create-slurm-cluster) have been met.

Configure Compute Fabric with TCP-X by setting the following environment variables for your environment.

```shell
NCCL_LIB_DIR='/var/lib/tcpxo/lib64' source /var/lib/tcpxo/lib64/nccl-env-profile.sh; \
	  export NCCL_FASTRAK_CTRL_DEV=enp0s12; \
	  export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0; \
	  export NCCL_SOCKET_IFNAME=enp0s12; \
	  export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices; \
	  export NCCL_NET=FasTrak; \
	  ls /var/lib/tcpxo/lib64;"
```

**Important:**

- This example has not been tested with the latest TCP-X version. Check with your cluster admin for the most recent instructions.
- If additional files need to be visible inside a running container, mount them explicitly with `RUN_CONF_MOUNTS` or place them in a path mounted by the recipe launch script.

### Azure

Azure requires two settings for optimal performance:

1. `NCCL_TOPO_FILE=<path to topo file under $LLMB_WORKLOAD>`
   - The VM topology files ensure that the correct CPUs, GPUs, and NICs are bound together. The location of this file varies, and it **must** be mounted into the container.
   - **Important:** The topology file must be visible inside the container. Mount it explicitly with `RUN_CONF_MOUNTS` or place it in a path mounted by the recipe launch script.
2. `NCCL_P2P_NET_CHUNKSIZE=2097152`
   - Increases the message size for NCCL send/recv for optimal performance.

Example configuration for a training recipe:

```shell
export NCCL_TOPO_FILE=$LLMB_WORKLOAD/nvd5-topo.xml # Exact location varies by cluster
export NCCL_P2P_NET_CHUNKSIZE=2097152
```

# Release Notes

For the latest updates, improvements, and breaking changes, see the [CHANGELOG](CHANGELOG).

# FAQ

This section summarizes common issues and resolutions.

## 1. Training logs contain multiple userbuffers.cu messages

### Symptom

Large-scale pre-training run logs contain messages like the following:

```
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 18 [2]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 18 [4]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 19 [2]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 19 [4]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 22 [2]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 22 [4]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 23 [2]: expecting 1 got 0
[userbuffers.cu:userbuffers_fp16_sum_inplace_gpu_rr_rs_oop_fp8:797] [6] Reduce-scatter: SM 23 [4]: expecting 1 got 0
```

### Solution

These messages usually indicate that one of the GPUs is hanging. Possible resolutions:

- Re-run the job on a different set of nodes.
- Reboot the affected nodes.

## 2. Slurm job failed or needs log inspection

### Symptom

A benchmark job failed or needs inspection.

### Solution

From `$LLMB_INSTALL`, list jobs and find the Slurm job ID:

```bash
cd $LLMB_INSTALL
llmb-run jobs
```

Then show the active log:

```bash
llmb-run jobs log <job_id>
```

By default, this prints the last 200 lines, not the full file. Use `--tail <lines>` for more lines, `--follow` for running jobs, or `--path` to print the active log file path.

If `llmb-run` cannot find the job, run `llmb-run jobs rebuild` once to scan older submissions, then retry the log command.

See the [llmb-run jobs command reference](cli/llmb-run/README.md#jobs-command) for the full command list and options.

## 3. NCCL InfiniBand QPS tuning

Some recipes set `NCCL_IB_QPS_PER_CONNECTION=4` by default. This controls the number of InfiniBand queue pairs NCCL uses per connection and can improve multi-node communication performance on certain cluster configurations.

If you need to set or override this value, there are two options:

**Option A** — Add it to the `environment` section of your `cluster_config.yaml` (applies to all jobs launched from that installation):

```yaml
environment:
  NCCL_IB_QPS_PER_CONNECTION: 4
```

**Option B** — Pass it inline when submitting a single job:

```bash
NCCL_IB_QPS_PER_CONNECTION=4 llmb-run submit -w <workload> -s <model-size> --dtype <precision> --scale <number>
```

> **Note:** The optimal value may vary by cluster and workload. If you experience communication errors or degraded performance after changing this setting, try removing it or adjusting the value.

## 4. Why do I see `Llama-3` downloads or `pretrain_llama3` log names when using the `llama3.1` recipe?

The `pretrain_llama3.1` workload is the user-facing recipe for 70B and 405B. Internally, the 70B size reuses the existing Megatron-Bridge `llama3` config instead of duplicating it under a separate `llama3.1` name. As a result, setup output for 70B may show `Meta-Llama-3-*`, and experiment or log names may use the `pretrain_llama3` prefix. This is expected and does not mean the wrong workload or model size was selected.

# Known Issues

## 1. Multiple GPU Types on Single Cluster

### Issue

The `llmb-install` tool currently supports only one GPU type per installation. If your cluster contains multiple GPU types (e.g., H100 and B200), you cannot install workloads for both GPU types in a single installation.

### Workaround

Create separate installations for each GPU type:

1. Run the installer once for your first GPU type (e.g., H100):

   ```bash
   ./install.sh
   # Select H100 workloads and specify an installation directory
   ```

2. Run the installer again for your second GPU type (e.g., B200):

   ```bash
   ./install.sh
   # Select B200 workloads and specify a different installation directory
   ```

Each installation will have its own `LLMB_INSTALL` directory. Use the appropriate `LLMB_INSTALL` directory for running workloads for each GPU type.

## 2. Cleanup-phase errors after successful run

### Issue

Some workloads complete all timesteps but print errors during the cleanup phase. This previously caused the Slurm job to be marked as failed.

### Workaround

We now detect this case and convert the exit code so Slurm reports success when the run actually finished. Log files will still contain the cleanup errors. If the job completed all timesteps and Slurm shows COMPLETED, you can ignore cleanup errors in the logs. This will be fixed in a future release.

## 3. AWS EFA limitations

### Issue

The following recipes have known AWS EFA limitations in this release:

- **Not supported on EFA**

  - DeepSeek V3 RL
  - Qwen3 30B on H100: the H100 configuration uses EP=16, which requires expert-parallel communication between nodes over EFA and exposes the Megatron-Bridge EP communication issue tracked in [Megatron-Bridge #3343](https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/3343).

- **Not validated on EFA**

  - DeepSeek V3 Megatron-Bridge on H100
  - Qwen3 235B on H100. Qwen3 235B is supported on GB300/GB200 systems.

## 4. Optional Priority Core Turbo binding for B300 Granite Rapids systems

### Issue

Some B300 Granite Rapids systems support Intel Priority Core Turbo (PCT). PCT is a turbo-frequency capability on some Intel Xeon 6900/6700-series Granite Rapids processors that lets a small number of high-priority CPU cores run at elevated turbo frequency while lower-priority cores run at a reduced frequency. On these systems, binding each local rank to the high-priority PCT cores can improve select Megatron-Bridge workloads.

PCT binding is disabled by default because LLMB does not currently detect PCT support or PCT core layout at runtime. Core layouts are platform-dependent; systems that enumerate CPU cores differently, such as round-robin enumeration on some Dell systems, may require a different binding than the B300 reference configuration.

Only enable this tuning after your cluster administrator confirms that PCT is supported and enabled. Also confirm whether your high-priority core IDs match the B300 reference binding used by Megatron-Bridge: `-C $((SLURM_LOCALID * 16)),$((SLURM_LOCALID * 16 + 1))`. Using the wrong binding can hurt performance or break recipes.

### Workaround

If your system matches the B300 reference PCT layout used by Megatron-Bridge, set `ENABLE_PCT_BINDING=true` when launching the recipe:

```bash
ENABLE_PCT_BINDING=true llmb-run submit -w pretrain_qwen3 -s 30b --dtype bf16 --scale 8
```

For B300 jobs, Megatron-Bridge then adds fixed-core PCT binding in addition to the default NUMA binding.

If your system uses a different PCT core layout, edit `common/b300_numa_cpu_pinning.patch` first and update the `-C ...` core binding expression to match your cluster. Each workload has its own Megatron-Bridge checkout, so apply the patch only to the workload you are testing.

```bash
cd $LLMB_INSTALL/workloads/pretrain_qwen3/Megatron-Bridge
git apply $LLMB_INSTALL/llmb_repo/common/b300_numa_cpu_pinning.patch
```

After applying the patch, submit the workload with `ENABLE_PCT_BINDING=true`.

## 5. KAI Scheduler components may run out of memory on large Kubernetes clusters

### Issue

On large Kubernetes clusters, [KAI Scheduler](https://github.com/kai-scheduler/KAI-Scheduler) components may be OOM-killed (out of memory). This is tracked in [KAI-Scheduler #1634](https://github.com/kai-scheduler/KAI-Scheduler/issues/1634). While a KAI Scheduler component is restarting, job pods can stay `Pending`.

### Workaround

Increase the memory (and, if needed, CPU) requests and limits of the affected KAI Scheduler components. See the linked issue for details.

## 6. Megatron-Bridge training hangs before the first iteration

### Issue

Megatron-Bridge workloads can hang in the first training step: no iteration is logged, and after the timeout the NCCL watchdogs fire on every node and the job exits. One rank is stuck waiting for PyTorch Inductor to compile fused kernels; Megatron-Core compiles them with `torch.compile`. On Python versions before 3.12.8 and on 3.13.0, Inductor's forked compile-worker pool can deadlock when it is re-created after being idle ([pytorch/pytorch#200848](https://github.com/pytorch/pytorch/issues/200848)). The hang has been seen with Llama 3.1 405B FP8 on B300 at 256 and 512 GPUs. Large runs with deep pipeline parallelism are the most exposed.

### Workaround

Set `TORCHINDUCTOR_COMPILE_THREADS=1` in the job environment. Inductor then compiles kernels in the training process instead of a worker pool. In partner testing on B300, the iteration time was unchanged.

```bash
llmb-run submit -w pretrain_llama3.1 -s 405b --dtype fp8 --scale 256 --env TORCHINDUCTOR_COMPILE_THREADS=1
```

When running a launch script directly, set `LLMB_MBRIDGE_EXTRA_ARGS="-E TORCHINDUCTOR_COMPILE_THREADS=1"`.

# Support

Terminology used in these recipes is explained in the [Appendix](APPENDIX.md).

For questions or to provide feedback, please contact LLMBenchmarks@nvidia.com.

# Notice and Disclaimer

This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.
