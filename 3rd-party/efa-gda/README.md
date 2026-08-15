# EFA Datapath Direct

This repository contains direct datapath implementations for Amazon's Elastic Fabric Adapter (EFA), enabling high-performance network operations with minimal CPU overhead.

## Elastic Fabric Adapter (EFA) Overview

### What is EFA?
Elastic Fabric Adapter (EFA) is Amazon's custom network interface designed for machine learning (ML) training, inference, and High Performance Computing (HPC) workloads on AWS. EFA provides:

- **High bandwidth networking**: Up to 400 Gbps network performance on latest instances
- **Low-latency communication**: Optimized for distributed ML training and inference
- **Bypass kernel networking**: Direct hardware access for improved performance
- **AWS integration**: Native support in AWS Nitro System architecture
- **ML framework optimization**: Optimized for PyTorch, TensorFlow, and other ML frameworks

### Scalable Reliable Datagram (SRD)
EFA uses SRD as its primary transport protocol, which provides:

- **Reliable delivery**: Guaranteed packet delivery with hardware-level acknowledgments
- **Multi-path load balancing**: Efficiently distributes traffic across multiple network paths
- **Fast failure recovery**: Quickly recovers from packet drops or link failures
- **High-throughput optimization**: Designed for bandwidth-intensive workloads
- **Hardware-accelerated congestion control**: Built-in flow control mechanisms

## EFA Datapath Implementations

### Traditional Implementations
1. **Kernel Driver**
   - Full kernel-space implementation
   - Standard verbs interface
   - Complete feature set with all EFA capabilities

2. **Userspace Libraries**
   - **libfabric provider**: Standard OFI (OpenFabrics Interface) implementation
   - **libibverbs provider**: RDMA verbs compatibility layer
   - **MPI libraries**: Direct integration with popular MPI implementations

### Direct Datapath Implementations
This repository focuses on **direct datapath** implementations that bypass traditional software stacks:

#### Current Implementation
- **[CUDA Datapath](./CUDA/)**: GPU-native EFA operations for CUDA applications
  - Direct posting of work requests from GPU kernels
  - GPU-side completion polling
  - No CPU involvement in data path operations
  - Optimized for GPU-to-GPU communication over EFA

#### Planned/Future Implementations
- **CPU Direct Path**: Userspace CPU implementation with direct hardware access
- **Additional accelerator support**: Support for other compute accelerators

## Use Cases

### Machine Learning and AI
- **Distributed ML training**: Large-scale model training across multiple GPUs and nodes
- **ML inference**: High-throughput inference serving with minimal latency
- **GPU-to-GPU communication**: Direct GPU communication for parameter synchronization
- **Model parallelism**: Efficient distribution of large models across multiple devices

### High-Performance Computing (HPC)
- **GPU-accelerated simulations**: Direct GPU-to-GPU communication
- **Scientific computing**: Large-scale parallel computations
- **Computational fluid dynamics**: High-bandwidth data exchange between compute nodes

### Performance-Critical Applications
- **Real-time analytics**: Low-latency data processing pipelines
- **Financial modeling**: High-frequency trading and risk calculations
- **Media processing**: Real-time video/audio processing workflows

## Getting Started

Each implementation directory contains its own detailed documentation:

- **[CUDA Implementation](./CUDA/README.md)**: Complete guide for GPU-based EFA operations
- Additional implementations will be documented as they are added

## Requirements

### Hardware
- EFA-enabled EC2 instances

### Software
- EFA kernel driver installed and configured
- Libibverbs (rdma-core) and EFA verbs provider
- Implementation-specific requirements (see individual directories)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

## Related Resources

- [Amazon EFA Documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [EFA Kernel Driver](https://github.com/amzn/amzn-drivers/tree/master/kernel/linux/efa)
- [Verbs EFA Provider](https://github.com/linux-rdma/rdma-core/tree/master/providers/efa)
