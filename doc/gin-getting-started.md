# Get started with GPU-Initiated Networking (GIN)

Plugin release 1.21 introduces support for NCCL's GPU-Initiated Networking (GIN) API. Following are instructions for getting started with GIN support.

Two modes of GIN are supported:

- Proxy mode: GPU queues network operations, with a CPU thread assisting with network communication.
- EFA-GDA (Elastic Fabric Adapter - GPU Direct Async): kernel directly initiates network communication, bypassing CPU entirely. Available on AWS EC2 with supported instance types (see [below](#efa-gda-elastic-fabric-adapter---gpu-direct-async)).

## Requirements

- [NCCL 2.31.2-1](https://github.com/NVIDIA/nccl/releases/tag/v2.31.2-1) or later
- [GDRCopy 2.5](https://github.com/NVIDIA/gdrcopy/releases/tag/v2.5) or later
- EFA-GDA additionally requires [Libfabric 2.6.0](https://github.com/ofiwg/libfabric/releases/tag/v2.6.0)+ and a supported EC2 instance type

## Testing

The GIN functionality can be tested (proxy mode) using the latest version of nccl-tests.

```bash
# Example to run the nccl-tests AllToAll benchmark using GIN kernels

# -R 2: enable symmetric memory registration -- required for GIN
# -D 3: selects the GIN alltoall device kernel
./alltoall_perf -R 2 -D 3
```

## EFA-GDA (Elastic Fabric Adapter - GPU Direct Async)

EFA ([Elastic Fabric Adapter](https://aws.amazon.com/hpc/efa/)) is a high-performance network interface for EC2, supporting low-latency communication at scale with OS bypass.

For supported AWS EC2 instance types with EFA, a fast path is supported for NCCL-GIN in which the GPU initiates network operations directly, without CPU involvement. Currently, P5en, P6-B200, and P6-B300 instance types are supported.

### Limitations

EFA-GDA has the following limitations:

1. Currently supports up to 512 signals+counters per EFA device
2. Does not support strong signals (signal delivery implies completion of previous puts) or VA signals (signals in registered virtual address; EFA only supports indexed signals). The application must set these requirements to false:

```
struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
reqs.ginStrongSignalsRequired = false;
reqs.ginVaSignalsRequired = false;
```

If an application needs functionality that is not currently supported by EFA-GDA, it can use the proxy mode as a fallback.

### Setup

To use EFA-GDA, two environment variables are required:

```bash
# Enable EFA-GDA explicitly; otherwise NCCL 2.31 defaults to proxy for EFA
export NCCL_GIN_TYPE=5
# Disable symmetric GIN kernels for NCCL collectives; these are not currently
# supported for EFA-GDA.
export NCCL_SYM_GIN_KERNELS_ENABLE=0
```

EFA-GDA functionality can be tested using the same nccl-tests AllToAll benchmark as above:

```bash
# Example to run the nccl-tests AllToAll benchmark using EFA-GDA GIN

export NCCL_GIN_TYPE=5
export NCCL_SYM_GIN_KERNELS_ENABLE=0

# -R 2: enable symmetric memory registration -- required for GIN
# -D 3: selects the GIN alltoall device kernel
# -V 2: reduce the number of CTAs used by the device kernel.
#       Required for EFA-GDA due to limitation on number of signals
./alltoall_perf -R 2 -D 3 -V 2
```

To confirm the EFA-GDA backend was used, set `NCCL_DEBUG=info` and `NCCL_DEBUG_SUSYS=INIT,ENV,NET` and look for the following log lines:
```
NCCL INFO NCCL_GIN_TYPE set by environment to 5.
...
NCCL INFO NET/OFI gin GDAKI: createContext done (...)
```
