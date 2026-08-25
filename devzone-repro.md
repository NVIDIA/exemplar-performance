# NVIDIA Deep Learning Inference Performance Reproduction Guide

This repository provides instructions reproduce inference performance data from the the [NVIDIA Deep Learning Performance - AI Inference](https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference) page.

## Prerequisites

Before configuring the orchestrator, ensure you have downloaded the required NVFP4 model weights from Hugging Face:

- **DeepSeek-R1 (DSR1):** [DeepSeek-R1-0528-NVFP4-v2](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2)
- **Qwen3.5-397B:** [Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- **Kimi-K2.5:** [Kimi-K2.5-NVFP4](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4)

## Environment Setup

Benchmarking is orchestrated using [srt-slurm](https://github.com/NVIDIA/srt-slurm), a command-line tool for distributed LLM inference benchmarks on SLURM clusters. (Support for benchmarking Kubernetes clusters coming soon.)

1. **Clone and Install:**

```bash
# Enter a directory on NFS, accessible by all nodes of your cluster.
git clone https://github.com/ishandhanani/srt-slurm.git 
cd srt-slurm
git checkout recipes/moe

# Initialize virtual environment and install dependencies (not shown)
uv venv
uv pip install -e .
```

2. **Initialize SLURM Workspace:**
   Execute the setup command below. You will be prompted to specify your SLURM account and partition.

```bash
#One-time setup (downloads NATS/ETCD, creates srtslurm.yaml)
make setup ARCH=aarch64  # or ARCH=x86_64
```

3. **Configure Model Paths:**
   The setup script will generate an `srtslurm.yaml` file. Edit this file to append your local model paths:

```yaml
model_path:
  dsr1: /path/to/local/dsr1
  qwen3.5-397b: /path/to/local/qwen3.5-397b
  kimi-k2.5: /path/to/local/kimi-k2.5
```

Depending on your cluster configuration, you may need to specify additional arguments in srtslurm.yaml. See https://github.com/ishandhanani/srt-slurm/blob/main/srtslurm.yaml.example for details.

## Running the Benchmarks

To execute a benchmark, apply the target configuration file using the `srtctl` CLI:

```bash
srtctl apply -f <path-to-config-file>
```

Available benchmarking configurations for published performance data are mapped below. Select the recipe that matches your target performance profile.

| Model            | 1K/1K                                                                                                                                                                              | 8K/1K                                                                                                                                                                                  | 1K/8K                                                                                             | 128K/8K                                                                                          |
| :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------- |
| **DSR1**         |                                                                                                                                                                                    | [GB300](https://github.com/ishandhanani/srt-slurm/blob/main/recipes/gb300-fp4/1k1k/max_tpt.yaml)                                                                                       | [GB300](https://github.com/ishandhanani/srt-slurm/tree/main/recipes/gb300-fp4/8k1k)               | [GB300](https://github.com/ishandhanani/srt-slurm/blob/main/recipes/gb300-fp4/1k8k/max-tpt.yaml) |
| **gpt-oss-120b** | [B200](https://github.com/ishandhanani/srt-slurm/tree/main/recipes/trtllm/b200-fp4/1k1k/mtp), [H200](https://github.com/ishandhanani/srt-slurm/tree/main/recipes/trtllm/h200/1k1k) | [B200](https://github.com/ishandhanani/srt-slurm/tree/main/recipes/trtllm/b200-fp4/8k1k/mtp), [H200](https://github.com/ishandhanani/srt-slurm/tree/main/recipes/trtllm/h200/8k1k/mtp) |                                                                                                   |                                                                                                  |
| **Kimi-K2.5**    | [B200](https://github.com/ishandhanani/srt-slurm/tree/recipes/moe/recipes/kimi-k2.5/b200/1k1k)                                                                                     | [B200](https://github.com/ishandhanani/srt-slurm/tree/recipes/moe/recipes/kimi-k2.5/b200/8k1k)                                                                                         | [B200](https://github.com/ishandhanani/srt-slurm/tree/recipes/moe/recipes/kimi-k2.5/b200/1k8k)    |                                                                                                  |
| **Qwen3.5-397B** | [B200](https://github.com/ishandhanani/srt-slurm/tree/recipes/moe/recipes/qwen3.5-397b/b200/1k1k)                                                                                  | [B200](https://github.com/ishandhanani/srt-slurm/tree/recipes/moe/recipes/qwen3.5-397b/b200/8k1k)                                                                                      | [B200](https://github.com/ishandhanani/srt-slurm/tree/recipes/moe/recipes/qwen3.5-397b/b200/1k8k) |                                                                                                  |

## Support

Terminology used in recipes is explained in the [Appendix](APPENDIX.md).

For questions or to provide feedback, please contact LLMBenchmarks@nvidia.com
