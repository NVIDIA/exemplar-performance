# Bulk Job Submission Examples

Note: Bulk mode is now accessed via `llmb-run submit -f <file>`.

## YAML Header Format

**Required format:** `workload_key_modelsize:`

The header must include both the workload name and model size, separated by an underscore.
The model size must end with `b` (billions of parameters) or `t` (trillions of parameters).

### Valid Examples

```yaml
pretrain_llama3.1_70b:      # Workload with decimal version
  tasks: [...]

pretrain_nemotron-h_56b:    # Workload with hyphen
  tasks: [...]

pretrain_deepseek-v3_671b:  # Large model
  tasks: [...]

pretrain_kimi-k2_1t:        # Trillion-parameter model
  tasks: [...]
```

### Invalid Examples

```yaml
pretrain_nemotron-h:        # ❌ Missing model size
  tasks: [...]

pretrain_llama_7x:          # ❌ Invalid format (must end with 'b' or 't')
  tasks: [...]
```

**Tip:** Use `llmb-run list` to see all available workload_modelsize combinations.

______________________________________________________________________

## Job Specification Formats

There are two supported file formats for file-based bulk job submission:

1. **YAML Format** (.yaml) - **Recommended**. Supports all features including environment variables and overrides.
2. **Text Format** (.txt) - **Legacy**. Supports basic configurations (workload, model size, dtype, scale, repeats).

### Choosing a Format

- Use **YAML** for almost all use cases. It is more readable and flexible.
- Use **Text** only if you need a quick-and-dirty list of basic jobs and prefer the compact tuple syntax.

## YAML Examples (Recommended)

### Basic Configuration

```yaml
pretrain_llama3.1_70b:
  tasks:
    - dtypes: 'fp8'
      scales: [64, 128, 256]
      repeats: 3
```

**Explanation**: This will run Llama 3.1 70B with fp8 precision at three different scales (64, 128, and 256 GPUs), repeating each configuration 3 times. Total of 9 jobs will be submitted.

### Multiple Data Types

```yaml
pretrain_llama3.1_70b:
  tasks:
    - dtypes: ['fp8', 'bf16']
      scales: [64, 128]
      repeats: 2
```

**Explanation**: This configuration will run the workload with both fp8 and bf16 precision at two different scales. Each combination (fp8@64, fp8@128, bf16@64, bf16@128) will be run twice. Total of 8 jobs will be submitted.

### With Proxy Configuration

```yaml
pretrain_deepseek-v3_671b:
  tasks:
    - dtypes: 'bf16'
      scales: [64]
      repeats: 1
      proxy: true    # Altered configuration for debug workflows
```

**Explanation**: This runs a proxy configuration - an altered config designed for debug workflows on fewer GPUs. Proxy results cannot be compared to production runs.

### With Environment Variables

```yaml
pretrain_qwen3_235b:
  defaults:
    env:
      DEBUG: true
  tasks:
    - dtypes: 'bf16'
      scales: [256, 512]
      repeats: 3
```

**Explanation**: This example sets global environment variables for all jobs. The workload will run with bf16 precision at two scales, with each configuration repeated 3 times. The environment variables will be applied to all 6 jobs.

### Complex Configuration (Overrides & Profiling)

```yaml
pretrain_llama3.1_405b:
  defaults:
    env:
      LOG_LEVEL: "INFO"
  dtypes: ['fp8', 'bf16']
  scales: [128, 256, 512]
  repeats: 2
  tasks:
    - scales: [128, 256]
      overrides:
        env:
          LOG_LEVEL: "DEBUG"
    - dtypes: ['fp8']
      scales: [512]
      repeats: 1
      profile: true
      overrides:
        env:
          LOG_LEVEL: "TRACE"
```

**Explanation**: This complex example demonstrates multiple features:

1. First task: Runs bf16 and fp8 at scales 128 and 256, with a `DEBUG` log level. Each configuration runs twice.
2. Second task: Overrides the dtype to only use fp8, and scale to 512. It is a single profiling run with `TRACE` log level.
3. All jobs will have the default `LOG_LEVEL` of `INFO` unless overridden.

### Multiple Workloads

```yaml
pretrain_llama3.1_405b:
  defaults:
    env:
      NCCL_IB_QPS_PER_CONNECTION: 1
  tasks:
    - dtypes: ['fp8', 'bf16']
      scales: [256, 512]
      repeats: 3

pretrain_qwen3_235b:
  tasks:
    - dtypes: 'bf16'
      scales: [256, 512]
      repeats: 2
```

## Text Format Examples (Legacy)

The text format uses a Python-tuple-like syntax:
`workload_modelsize: (dtype, [scales], repeats)`

**Important**: Inline trailing comments on task lines are **not supported**. Put comments on their own lines instead.

### Basic Text Example

```
pretrain_llama3.1_405b:
('bf16', [128, 256], 1)
('fp8', [128, 256, 512], 3)
```

### Mixed Text Example

```
pretrain_qwen3_235b:
('bf16', [256, 512], 2)
# True enables profiling
('bf16', [512], 1, True)
```

**Note**: The example above shows the correct way to add comments - on their own lines. Inline comments like `('fp8', [512], 1, True)  # comment` will cause parsing errors.

## Usage

To run any of these examples:

```bash
# Using the unified submit command
llmb-run submit -f my_config.yaml

# Dry run to preview jobs (recommended first step)
llmb-run submit -f my_config.yaml --dry-run
```

## Troubleshooting

### "No tasks generated" Error

If you see `ERROR: No tasks generated`, ensure your YAML includes at least one task entry:

❌ **Incorrect** (generates no tasks):

```yaml
pretrain_llama3.1_70b:
  dtypes: fp8
  scales: [128, 256]
  tasks: []
```

✅ **Correct** (specify tasks directly):

```yaml
pretrain_llama3.1_70b:
  tasks:
    - dtypes: fp8
      scales: [128, 256]
```

**Note**: Top-level `dtypes`/`scales` are defaults for task inheritance, not task definitions. Always define jobs under `tasks:`.

## Notes

- All examples assume a valid `cluster_config.yaml` file is present.
- The `repeats` parameter defaults to 1 if not specified.
- To run a profiling job in YAML, create a task with `profile: true`.
- Use the `--dry-run` flag to preview all jobs before submission.
