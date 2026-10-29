# Spectrum-X Reference Architectures

NVIDIA Spectrum-X provides an Ethernet RoCE compute fabric for communication
between GPUs across nodes. NVLink provides communication within an NVLink
domain. On GB200 and GB300 NVL72 systems, that domain spans multiple compute
trays; allocating multiple Slurm nodes alone does not demonstrate traffic across
the scale-out fabric.

The references below describe Spectrum-X configurations for the GPU platforms
covered by Exemplar Performance. They are architecture guidance, not a new set of
measured Exemplar baselines. Existing recipe performance tables retain the
provenance of their original runs.

## Platform references

| GPU platform | Spectrum-X compute fabric reference | Design source |
| --- | --- | --- |
| HGX H100 / B200 | Eight 400 GbE BlueField-3 B3140H SuperNICs per eight-GPU server, with Spectrum SN5600/SN5610 switches | [HGX H100/H200/B200 Enterprise RA](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/network-logical-architecture.html) |
| GB200 NVL72 | ConnectX-7 NICs in a single-plane Spectrum-X configuration; confirm adapter count and active port speeds for the deployed tray | [Spectrum-X architecture and NIC mapping](https://docs.nvidia.com/networking/display/kubernetes2670/spectrum-x/spectrum-x.html) |
| HGX B300 | Eight ConnectX-8 SuperNICs per eight-GPU server, each providing 800 Gb/s through two 400 GbE links in the dual-plane design, with Spectrum SN5600 switches | [HGX B300 Enterprise RA](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/network-logical-architecture.html) |
| GB300 NVL72 | Four ConnectX-8 SuperNICs per four-GPU tray, providing 3.2 Tb/s aggregate compute bandwidth per tray in the dual-plane design | [GB300 NVL72 Enterprise RA](https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/network-logical-architecture.html) |

These are specific reference designs. Use the RA that matches the installed
hardware and software release when choosing adapters, link speeds, plane layout,
and topology. The HGX references do not imply that an existing DGX InfiniBand
baseline was measured on the corresponding HGX Ethernet configuration.

## Fabric and software prerequisites

Use the supported switch software, NIC firmware, host networking stack, GPU
driver, and NCCL/plugin combination for the selected Spectrum-X release. The
generic minimum versions in the main README do not establish compatibility for
every Spectrum-X configuration.

Configure compute RoCE addressing, routing, MTU, congestion control, and QoS using
the selected RA. Record rails and planes separately: a software multiplane
configuration exposes planes to the application, while hardware multiplane
handles plane selection in hardware. See the
[Spectrum-X solution documentation](https://www.nvidia.com/en-us/networking/ethernet/spectrum-x/)
and the deployment documentation for your release.

The NCCL benchmark container must include compatible NCCL, MPI-enabled
`nccl-tests`, and the Spectrum-X NCCL plugin for the node CPU architecture.
Follow the [HPC-X Spectrum-X plugin instructions](https://networking-docs.nvidia.com/hpcxum/2.51/spectrum-x-nccl-plugin)
for plugin loading and library paths. Check the per-rank NCCL NET logs to verify
that the intended plugin and RoCE devices were selected.

## Validate the compute fabric

Run the [NCCL microbenchmark](../../microbenchmarks/nccl/README.md#spectrum-x-nccl-profile)
with its opt-in Spectrum-X profile before comparing application performance.
The recipe currently supports Slurm; the Kubernetes deployment references above
provide fabric guidance and do not add Kubernetes support to this recipe.

1. Run the full communicator cases to measure the configured combination of
   NVLink and the scale-out fabric.
2. Run the split cases, which place one GPU from each node in each communicator.
   For NVL72, MNNVL is disabled in those cases to exercise the network between
   trays. Verify rank placement and communicator membership in the logs.
3. Inspect correctness results, out-of-place bus bandwidth, and latency at each
   message size. Split-test bandwidth is reported per communicator; retain that
   distinction when reporting or aggregating results.
4. Run the NoNVLink cases to exercise the fabric with MNNVL, P2P, SHM and PXN
   disabled. These use cyclic rank placement; verify it independently from the
   block placement used for Mask cases.

Fabric bandwidth depends on adapter speeds, active planes, oversubscription,
rank placement, and the collective. Establish baselines for that configuration
before assigning performance pass thresholds. The initial profile does not
establish Exemplar qualification criteria or validate isolation and failover.

## Record each benchmark configuration

Include the following with submitted results so a comparison can be reproduced:

| Area | Details |
| --- | --- |
| Compute | GPU platform, GPUs per node, node count, participating GPUs, CPU architecture, and NVLink domain layout |
| Fabric | RA version, switch model/software, NIC model/firmware, adapter and active port counts, negotiated speeds, rails, planes, and topology/oversubscription |
| Host | OS/kernel, GPU driver, networking driver/DOCA-OFED, GPUDirect RDMA mode, MTU, RoCE addressing, and QoS |
| Container | Image digest, CUDA, MPI, NCCL, Spectrum-X plugin version, and nccl-tests revision |
| Test | Exemplar revision, generated launcher and TOML files, NCCL environment, message sizes, registration mode, communicator split, and rank/node placement |
| Results | Correctness, bandwidth/latency by size and collective, NCCL NET logs, and any concurrent traffic or degraded links |

Keep management and storage network specifications separate from the GPU
compute fabric. A system's Ethernet storage interfaces do not establish that its
NCCL compute traffic used Spectrum-X.
