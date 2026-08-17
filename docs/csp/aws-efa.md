# AWS EFA container guidance

The NeMo images used by the Performance Recipes already include Elastic Fabric Adapter (EFA) support. Update the EFA software only when troubleshooting a communication or performance issue.

## Verify that EFA is active

First, confirm that EFA is available on an allocated compute node:

```bash
/opt/amazon/efa/bin/fi_info -p efa
```

The command should list one or more EFA interfaces. If it does not, resolve the host configuration before changing the container.

Next, submit a representative multi-node job with NCCL initialization and network logging enabled:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
  llmb-run submit -w <workload> -s <model-size> --dtype <precision> --scale <number>
```

Set `JOB_ID` to the submitted Slurm job ID, then check the job log for the network NCCL selected. NCCL prints this during initialization, near the start of the log, so search the whole file rather than the default tail:

```bash
JOB_ID=12345
grep -E "Using network|Selected provider|net plugin" "$(llmb-run jobs log "$JOB_ID" --path)"
```

Healthy output names Libfabric and the EFA provider. Exact wording and device counts vary by NCCL and plugin version; on a working node it looks similar to:

```
NCCL INFO Loaded net plugin Libfabric (v9)
NCCL INFO Using network Libfabric
NCCL INFO Selected provider is efa (found 16 nics)
```

`Using network Socket`, or an error loading the network plugin, means NCCL is not using EFA for inter-node communication. Check the host EFA devices and the container's library paths before rebuilding the image.

Turn off `NCCL_DEBUG` once EFA is confirmed. It adds significant log volume at scale, so leave it out of the runs you are measuring.

## Test an updated EFA software stack

The container's EFA stack can lag the host, since EFA updates reach the NeMo images after the upstream NVIDIA base containers.

If diagnostics point to the container, build a new image from the same NeMo image tag used by the recipe. Use the actively maintained [AWS NCCL tests Dockerfile](https://github.com/awslabs/awsome-distributed-ai/blob/main/micro-benchmarks/nccl-tests/nccl-tests.Dockerfile) as the source for current EFA installer, `aws-ofi-nccl`, and GDRCopy versions and installation steps.

Do not carry forward EFA tuning variables from older images or guides unless the current AWS reference requires them. Current EFA installers use the OFI Tuner plugin to configure settings previously controlled by variables such as `FI_USE_HUGE_PAGE` and `FI_EFA_DEVICE_RDMA`.

The reference Dockerfile starts from a CUDA development image, so it is not a replacement for a Performance Recipe image.

Keep the NCCL version included in the NeMo image for the initial test. If updating EFA does not resolve the issue, test the NCCL version from the AWS reference as a separate change.

Find the image for the recipe you are testing. Recipes do not all pin the same NeMo version, so check the recipe rather than assuming. `<recipe>/metadata.yaml` is authoritative and lists the image under `container.images`:

```yaml
container:
  images:
    - 'nvcr.io#nvidia/nemo:26.04.01'
```

The `#` is an Enroot separator, equivalent to `/`. `llmb-install` stores that image as `$LLMB_INSTALL/images/nvidia+nemo+26.04.01.sqsh`. The recipe's `launch.sh` sets the same tag as `FW_VERSION`.

### Build with Docker

Create a derived Dockerfile that starts from the recipe image and applies only the EFA installation steps:

```dockerfile
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Copy the EFA installer and GDRCopy stanzas from the AWS reference Dockerfile here,
# using the versions it pins. Keep the NCCL supplied by the NeMo base image, and
# skip the reference Dockerfile's CUDA and NCCL-tests stanzas.
```

Build it against the recipe's image tag:

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/nemo:<version> \
  -t llmb-nemo-efa:local .
```

Convert the image to SquashFS for Enroot/Pyxis:

```bash
enroot import \
  -o "$LLMB_INSTALL/images/nvidia+nemo+<version>-efa.sqsh" \
  dockerd://llmb-nemo-efa:local
```

### Build with Pyxis

Docker is often unavailable on HPC login nodes. Pyxis can instead start a writable container on an allocated compute node and save the modified filesystem when the job exits.

> **Note:** `--container-save` overwrites an existing target without prompting. Use a new filename, such as `nvidia+nemo+<version>-efa.sqsh`, or confirm that the target can be replaced.

```bash
srun --nodes=1 --ntasks=1 <allocation-options> \
  --container-image="$LLMB_INSTALL/images/nvidia+nemo+<version>.sqsh" \
  --container-writable \
  --container-remap-root \
  --no-container-mount-home \
  --container-save="$LLMB_INSTALL/images/nvidia+nemo+<version>-efa.sqsh" \
  --pty bash
```

Inside the container, run the same installation steps, then exit. Pyxis exports the modified container to the `--container-save` path after the job completes. Two things to plan for:

- The output directory must exist and have room for the image, which is tens of gigabytes.
- The allocation must outlast the final SquashFS export. If the job is interrupted before the export finishes, the updated image is not saved.

The node also needs network access to download the EFA packages.

## Test the updated image

Test the new image without replacing the image installed by `llmb-install`:

```bash
RUN_CONF_IMAGE="$LLMB_INSTALL/images/nvidia+nemo+<version>-efa.sqsh" \
  llmb-run submit -w <workload> -s <model-size> --dtype <precision> --scale <number>
```

Repeat the EFA verification with the updated image before comparing performance. Rebuild the derived image whenever the recipe's base image changes rather than carrying an older one forward.

### Use the updated image for repeated runs

`RUN_CONF_IMAGE` is convenient for initial testing or an installation that runs only one recipe. For repeated use, back up the original image and symlink the updated image to the filename expected by the recipe:

```bash
ORIGINAL_IMAGE="$LLMB_INSTALL/images/nvidia+nemo+<version>.sqsh"
UPDATED_IMAGE="$LLMB_INSTALL/images/nvidia+nemo+<version>-efa.sqsh"
BACKUP_IMAGE="${ORIGINAL_IMAGE%.sqsh}.original.sqsh"

mv -i "$ORIGINAL_IMAGE" "$BACKUP_IMAGE"
ln -s "$UPDATED_IMAGE" "$ORIGINAL_IMAGE"
```

Repeat this for each distinct NeMo image version you update. Do not point a recipe at an updated image derived from a different NeMo version. To restore the original image, remove the symlink and move the backup to its original filename.
