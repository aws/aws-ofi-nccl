# AWS OFI NCCL

AWS OFI NCCL is a plugin which lets EC2 customers use [libfabric](https://github.com/ofiwg/libfabric) as a network provider while running [NCCL](https://github.com/NVIDIA/nccl) based applications.

## Overview

Machine learning frameworks running on top of NVIDIA GPUs use a library called [NCCL](https://developer.nvidia.com/nccl) which provides standard collective communication routines for an arbitrary number of GPUs installed across single or multiple nodes. Currently, NCCL uses InfiniBand Verbs or TCP/IP sockets for transport layer communication.

This project implements a plugin which maps NCCLs connection-oriented transport APIs to [libfabric's](https://ofiwg.github.io/libfabric/) connection-less reliable interface. This allows NCCL applications to take benefit of libfabric's transport layer services like RDMA or reliable message support, and directly access network resources without operating system interventions.

## Requirements

The plugin currently supports Ubuntu and Redhat-based operating systems, Libfabric v1.6 and NCCL v2.0.

Libfabric API implementation supports various providers. The plugin can choose only those which support the following features  as defined in the [libfabric API documentation](https://ofiwg.github.io/libfabric/v1.6.1/man/).

* Tagged messaging (`FI_TAGGED`, `FI_MSG`)
* Source address availability in completions (`FI_SOURCE`)
* Data transfer context structures (`FI_CONTEXT`)
* Reliable datagram endpoints (`FI_RDM`)
* Send after Send ordering semantics (`FI_ORDER_SAS`)
* Indexed based addressing in address vectors (`FI_AV_TABLE`)
* Automatic control and data progress model (`FI_PROGRESS_AUTO`)

## Getting Started

### Dependencies

`aws-ofi-nccl` requires working installations of NCCL, libfabric and MPI. You can find the instructions for installing the first two at [NCCL installation](https://github.com/NVIDIA/nccl) and [libfabric installation](https://github.com/ofiwg/libfabric) respectively. To install MPI, you can use standard packages provided for your linux distribution.

### Build Instructions

1. First, clone this repository by running

```
git clone git@github.com:aws/aws-ofi-nccl.git
```

2. Then, compile the plugin by running
```
cd aws-ofi-nccl
make
```

**Note:** The Makefile assumes the following paths.
```
NCCL_HOME=/opt/nccl
CUDA_HOME=/usr/local/cuda
OFI_HOME=/opt/libfabric
```

If your installation paths are different, please ensure to define that while running `make`, as below.
```
make NCCL_HOME=/path/to/nccl/ CUDA_HOME=/path/to/cuda OFI_HOME=/path/to/ofi
```

### Running Unit Tests

Running unit tests requires a working [MPI setup](https://www.open-mpi.org/faq/?category=running) between the communicating hosts. Once MPI is setup, you can use commands like below for running any test of your choice.

```
mpirun -n 2 --host <host-1>,<host-2> -tag-output ./tests/nccl_message_transfer
```

**Note:** All tests require 2 hosts to run except [ring_c.c](path_to_ring_c.c) which requires atleast 3 hosts.

### Running nccl-perf tests

To run standard `nccl-perf` tests with the `aws-ofi-nccl` plugin, you can follow the instructions below.

1. Clone the repository
```
git clone https://github.com/NVIDIA/nccl-tests.git
```

2. Build the tests
```
cd  nccl-tests/
make NCCL_HOME=~/nccl/build/
```

3. Run perf tests
```
NCCL_DEBUG=INFO LD_LIBRARY_PATH=/path/to/libfabric:/path/to/libfabric/lib:/path/to/nccl/build/lib:/path/to/aws-ofi-nccl/:$LD_LIBRARY_PATH build/all_reduce_perf -b 8 -f 2 -e 32M -c 1
```

## Known Issues

The plugin returns only 1 NIC device even if the system supports multiple NICs.

## Getting Help

If you have any issues in building or using the package or if you think you may have found a bug, please open an [issue](https://github.com/aws/aws-ofi-nccl/issues).

## Contributing

Reporting issues and sending pull requests are always welcome. To learn how you can contribute, please look at our [contributing guidelines](CONTRIBUTING.md#contributing-guidelines).

## License

This library is licensed under the [Apache 2.0 License](LICENSE). 
