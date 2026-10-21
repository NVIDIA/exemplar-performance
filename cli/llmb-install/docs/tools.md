# Workload-Specific Tools

> **For Recipe Developers**: This guide shows how to configure workload-specific tool versions in your metadata.yaml files.
>
> **See Also**: [Recipe Development Guide](recipe_guide.md) for complete metadata.yaml documentation.

## Overview

The `tools` section in metadata.yaml allows you to specify tool versions (like nsys) that will be automatically downloaded and installed for your workload.

## When to Use

**Only use the tools section when you need a specific tool version.** If your container's tools work fine, don't add this section.

Common use cases:

- Specific nsys version required for profiling on certain GPU types
- Different GPU types need different tool versions
- Container's tool version has bugs or missing features

## Supported Tools

Currently supported:

- `nsys` - NVIDIA Nsight Systems profiler
- `cuda_cupti_lib` - CUDA CUPTI library (profiling workaround)

## Format Options

### Option 1: Simple Format (All GPUs Same Version)

Use this when all GPU types need the same tool version:

```yaml
tools:
  nsys: "2025.5.1.120"
```

### Option 2: GPU-Conditional (Different Versions Per GPU)

Use this when different GPU types need different versions:

```yaml
tools:
  nsys:
    by_gpu:
      h100: "2025.1.1"
      gb300: "2025.6.0"
      default: "2025.5.1.120"  # Optional: fallback for other GPUs
```

### Option 3: Partial Coverage (Some GPUs Use Container)

Use this when only specific GPUs need a custom version:

```yaml
tools:
  nsys:
    by_gpu:
      h100: "2025.1.1"
      gb300: "2025.6.0"
      # b200, gb200 not listed = use container version
```

## Complete Examples

### Example 1: All GPUs Need Same nsys Version

```yaml
general:
  workload: my_workload
  workload_type: pretrain
  framework: nemo2

container:
  images: 
    - 'nvcr.io#nvidia/nemo:25.07.01'

tools:
  nsys: "2025.5.1.120"

run:
  launcher_type: 'nemo'
  launch_script: 'launch.sh'
  gpu_configs:
    h100:
      model_configs:
        - model_size: '405b'
          dtypes: ['fp8']
          scales: [512]
    b200:
      model_configs:
        - model_size: '405b'
          dtypes: ['fp8']
          scales: [256]
```

**Result**:

- h100: Downloads nsys 2025.5.1.120
- b200: Downloads nsys 2025.5.1.120

### Example 2: Different Versions for Different GPUs

```yaml
general:
  workload: my_workload
  workload_type: pretrain
  framework: nemo2

container:
  images: 
    - 'nvcr.io#nvidia/nemo:25.07.01'

tools:
  nsys:
    by_gpu:
      h100: "2025.1.1"
      gb300: "2025.6.0"
      default: "2025.5.1.120"

run:
  launcher_type: 'nemo'
  launch_script: 'launch.sh'
  gpu_configs:
    h100:
      model_configs: [...]
    gb300:
      model_configs: [...]
    b200:
      model_configs: [...]
```

**Result**:

- h100: Downloads nsys 2025.1.1
- gb300: Downloads nsys 2025.6.0
- b200: Downloads nsys 2025.5.1.120 (default)

### Example 3: Only Some GPUs Need Custom Version

```yaml
general:
  workload: my_workload
  workload_type: pretrain
  framework: nemo2

container:
  images: 
    - 'nvcr.io#nvidia/nemo:25.07.01'

tools:
  nsys:
    by_gpu:
      h100: "2025.1.1"
      gb300: "2025.6.0"
      # No default - other GPUs use container version

run:
  launcher_type: 'nemo'
  launch_script: 'launch.sh'
  gpu_configs:
    h100:
      model_configs: [...]
    gb300:
      model_configs: [...]
    b200:
      model_configs: [...]
    gb200:
      model_configs: [...]
```

**Result**:

- h100: Downloads nsys 2025.1.1
- gb300: Downloads nsys 2025.6.0
- b200: Uses container nsys (no download)
- gb200: Uses container nsys (no download)

## Installation Behavior

### Directory Structure

Tools are installed to:

```text
$LLMB_INSTALL/tools/{tool_name}/{version}/
```

For example:

```text
$LLMB_INSTALL/tools/nsys/2025.1.1/bin/nsys
$LLMB_INSTALL/tools/nsys/2025.5.1.120/bin/nsys
```

### Deduplication

If multiple workloads require the same tool version, it's only downloaded once:

```yaml
# Workload A
tools:
  nsys: "2025.1.1"

# Workload B  
tools:
  nsys: "2025.1.1"
```

Result: nsys 2025.1.1 downloaded only once, shared by both workloads.

### Skip Existing

If you re-run installation and a tool/version already exists, it's skipped:

```text
Skipping nsys 2025.1.1 -- already installed at $LLMB_INSTALL/tools/nsys/2025.1.1
```

## Resolution Logic

The tool resolution follows this priority:

1. **GPU explicitly listed** in `by_gpu` → use that version
2. **`default` key exists** → use default version
3. **Neither exists** → use container version (no download)

## Tips and Best Practices

1. **Only specify when needed**: If container version works, don't add tools section
2. **Use `default` carefully**: Only add if you want all unlisted GPUs to download a version
3. **Version format**: Use exact version strings as they appear in nsys releases (e.g., "2025.1.1", "2025.5.1.120")
4. **Check supported GPUs**: Use `vr`, `gb300`, `gb200`, `b300`, `b200`, `h100`, or `default`

## Troubleshooting

### Tool not downloading

- Check that the tool is in the `by_gpu` list or `default` is set
- Verify GPU type spelling matches supported types

### Wrong version installed

- Check that GPU type is spelled correctly in `by_gpu`
- Verify version string matches exactly

### Download fails

- Verify version string is valid for the tool
- Check network connectivity and download URL accessibility
- Ensure the login node has internet access

## CUDA CUPTI Library

The `cuda_cupti_lib` tool provides an alternative profiling workaround by mounting a specific CUPTI library version into containers. Use this when nsys profiling requires a different CUPTI version than what's in the container.

### When to Use cuda_cupti_lib

- Nsys profiling fails due to CUPTI version mismatch
- Container's CUPTI library is incompatible with profiling workload
- Simpler alternative to replacing entire nsys installation

### Example Usage

```yaml
tools:
  cuda_cupti_lib: "13.0.85"
```

Or with GPU-conditional versioning:

```yaml
tools:
  cuda_cupti_lib:
    by_gpu:
      h100: "13.0.85"
      gb300: "13.0.85"
```

### Installation Result

Extracts to `$LLMB_INSTALL/tools/cuda_cupti_lib/{version}/lib/` with files like:

- `libcupti.so.2025.3.1` (versioned library)
- `libcupti.so` (symlink)

During profiling runs, llmb-run mounts the versioned library into the container at the appropriate location for your architecture.

## Version Information

### NSys Versions

Check the [NVIDIA Nsight Systems download page](https://developer.nvidia.com/nsight-systems) for available versions.

Version format: `YYYY.MM.PATCH.BUILD-COMMIT`

Examples:

- `2025.1.1.118-3638078`
- `2025.5.1.121-3638078`

### CUDA CUPTI Versions

CUPTI libraries are distributed as part of CUDA toolkit redistributables.

Version format: `MAJOR.MINOR.PATCH`

Examples:

- `13.0.85`
- `12.6.77`
