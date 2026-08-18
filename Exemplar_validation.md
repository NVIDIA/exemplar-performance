# **NVIDIA Exemplar Cloud – Path & Requirements**

For cloud partners using this repo to work toward [NVIDIA Exemplar Cloud status](https://www.nvidia.com/en-us/data-center/ai-cloud-performance/).

## **Exemplar Performance**

Benchmarking your cloud's performance across a suite of real AI workloads ensures consistent, high-quality GPU performance for end users.

Validating the performance of your platform is a transparent way to demonstrate that your unique cloud delivers cutting-edge throughput and provides excellent perf/TCO value.

There are two common usage patterns for this repo:

- Achieving NVIDIA-validated Exemplar performance by demonstrating > 95% baseline performance for all workloads in a given test suite.
- Using Recipes ad hoc independently for hardware, cluster, or software performance checks, e.g. during maintenance windows or during new hardware bring-up / NPI.

### About Exemplar

- Exemplar performance is validated per chip, with test suites available for GB300, GB200, B300, B200, and H100 chips.
- Benchmarking scripts ("Recipes") are released every ~2 months, evolving as common AI workload patterns evolve and new models ship. You can find Recipe versions as tags/releases in this repo.

## **Stages of Exemplar**

### **Kickoff with NVIDIA**

While the benchmarks can be run independently, we recommend looping in your NVIDIA account team before you begin. We have performance experts available to help with questions about the test suite and to help investigate tuning opportunities as needed.

### **Prepare a benchmark cluster**

1. At this time, Exemplar tests require Slurm clusters. Instructions for system and cluster requirements can be found on the main [README](README.md) of this repo.
2. You must use the latest version of the recipes available at the time you begin testing and continue to use that same version for the entirety of your exemplar certification process.
3. During install, you will be prompted for workload selection. If you select 'Exemplar Cloud', the full Exemplar test suite for your selected GPU type will be installed.
4. Validate the system by running the [prescreen test](microbenchmarks/system_info/README.md).

### **Run benchmark recipes via llmb-run**

```bash
llmb-run exemplar
```

This launches the full Exemplar test suite for the installed GPU type. `llmb-run` is the recommended tool for executing the suite; the `exemplar` subcommand is a convenience that launches every required workload in one go.

If individual workloads fail, you can re-run them on their own — Exemplar requires a passing run for each workload in the suite, not a single end-to-end execution. See the [llmb-run README](cli/llmb-run/README.md) for repeat and profiling behavior.

### **Verify results**

1. Package your results for submission using `llmb-run archive`
2. This creates a compressed `.tar.zst` file under `$LLMB_INSTALL/` containing all experiment logs and configuration metadata. Profiling data is excluded to keep the archive compact — share profiles separately if requested. See the [llmb-run README](cli/llmb-run/README.md) for options.
3. Submit the archive to your NVIDIA account team for review.

### **Optimize with NVIDIA**

Work with your NVIDIA account team to investigate any tuning opportunities with NVIDIA performance experts.

### **Qualify for Exemplar**

If approved, your cloud is recognized as an [NVIDIA Exemplar Cloud](https://www.nvidia.com/en-us/data-center/ai-cloud-performance/) for the selected platform(s). NVIDIA is happy to collaborate to support downstream efforts highlighting your achievement.

## **Ongoing Expectations**

- Periodically re-run recipes and maintain performance vs. updated baselines to ensure the platform is delivering optimal perf/value for end users.
- An Exemplar validation from NVIDIA is valid for 12 months.

To start, contact your NVIDIA account team and reference this Exemplar Performance repo.

## Exemplar Workload Recipes

Scale: **512 GPUs** | Repeats: **1** | Profiling: **disabled**

### GB300

| Model       | Size | Dtypes           |
| :---------- | :--- | :--------------- |
| DeepSeek-V3 | 671B | BF16, FP8, NVFP4 |
| GPT (OSS)   | 120B | BF16             |
| Kimi-K2     | 1T   | FP8              |
| Llama 3.1   | 405B | FP8, NVFP4       |
| Llama 3.1   | 70B  | FP8, NVFP4       |
| Nemotron-H  | 56B  | FP8              |
| Nemotron 3  | 120B | BF16, FP8        |
| Qwen3       | 235B | BF16, FP8        |

### GB200

| Model       | Size | Dtypes           |
| :---------- | :--- | :--------------- |
| DeepSeek-V3 | 671B | BF16, FP8, NVFP4 |
| GPT (OSS)   | 120B | BF16             |
| Kimi-K2     | 1T   | FP8              |
| Llama 3.1   | 405B | FP8, NVFP4       |
| Llama 3.1   | 70B  | FP8, NVFP4       |
| Nemotron-H  | 56B  | FP8              |
| Nemotron 3  | 120B | BF16             |
| Qwen3       | 235B | BF16, FP8        |

### B300

| Model       | Size | Dtypes           |
| :---------- | :--- | :--------------- |
| DeepSeek-V3 | 671B | BF16, FP8, NVFP4 |
| GPT (OSS)   | 120B | BF16             |
| Kimi-K2     | 1T   | FP8              |
| Llama 3.1   | 405B | FP8, NVFP4       |
| Llama 3.1   | 70B  | FP8, NVFP4       |
| Nemotron-H  | 56B  | FP8              |
| Nemotron 3  | 120B | BF16             |
| Qwen3       | 235B | BF16, FP8        |

### B200

| Model       | Size | Dtypes     |
| :---------- | :--- | :--------- |
| DeepSeek-V3 | 671B | BF16, FP8  |
| GPT (OSS)   | 120B | BF16       |
| Kimi-K2     | 1T   | FP8        |
| Llama 3.1   | 405B | FP8, NVFP4 |
| Llama 3.1   | 70B  | FP8, NVFP4 |
| Nemotron-H  | 56B  | FP8        |
| Nemotron 3  | 120B | BF16       |
| Qwen3       | 235B | BF16, FP8  |

### H100

| Model      | Size | Dtypes    |
| :--------- | :--- | :-------- |
| GPT (OSS)  | 120B | BF16      |
| Llama 3.1  | 70B  | BF16, FP8 |
| Nemotron-H | 56B  | FP8       |
| Qwen3      | 235B | BF16      |
