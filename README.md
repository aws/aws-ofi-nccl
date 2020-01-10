*NOTE: This is an experimental branch specifically targeted for testing on AWS.
Therefore, This branch is not supported.*

# AWS OFI NCCL

AWS OFI NCCL is a plug-in which enables EC2 developers to use
[libfabric](https://github.com/ofiwg/libfabric) as a network provider while
running [NVIDIA's NCCL](https://github.com/NVIDIA/nccl) based applications.

## Overview

Machine learning frameworks running on top of NVIDIA GPUs use a library called
[NCCL](https://developer.nvidia.com/nccl) which provides standard collective
communication routines for an arbitrary number of GPUs installed across single
or multiple nodes.

This project implements a plug-in which maps NCCLs connection-oriented
transport APIs to [libfabric's](https://ofiwg.github.io/libfabric/)
connection-less reliable interface. This allows NCCL applications to take
benefit of libfabric's transport layer services like reliable message support
and operating system bypass.

## Requirements

The plug-in currently supports the following distributions:
* Amazon Linux
* Amazon Linux 2
* Redhat Enterprise Linux 7
* Ubuntu 16.04 LTS
* CentOS 7

It also requires
[Libfabric v1.9.x](https://github.com/ofiwg/libfabric/tree/v1.9.x)
and supports [NCCL v2.5.6](https://github.com/NVIDIA/nccl/releases/tag/v2.5.6-2).

Libfabric supports various providers. The plug-in can choose only those which
support the following features as defined in the
[libfabric API documentation](https://github.com/ofiwg/libfabric/tree/master/man/).

* Tagged messaging (`FI_TAGGED`, `FI_MSG`)
* Data transfer context structures (`FI_CONTEXT`)
* Reliable datagram endpoints (`FI_EP_RDM`)
* Send after Send ordering semantics (`FI_ORDER_SAS`)
* Automatic control and data progress model (`FI_PROGRESS_AUTO`)

## Getting Started

### Dependencies

`aws-ofi-nccl` requires working installations of NCCL and libfabric. You can
find the instructions for installing the first two at
[NCCL installation](https://github.com/NVIDIA/nccl) and
[libfabric installation](https://github.com/ofiwg/libfabric) respectively.

### Build Instructions

The plugin uses GNU autotools for its build system. You can build it as follows:

```
$ ./autogen.sh
$ ./configure
$ make
$ sudo make install
```

If you want to install the plugin in a custom path, use the `--prefix`
configure flag to provide the path. You can also point the build to custom
dependencies with the following flags:

```
  --with-libfabric=PATH   Path to non-standard libfabric installation
  --with-cuda=PATH        Path to non-standard CUDA installation
  --with-nccl=PATH        Path to non-standard NCCL installation
  --with-mpi=PATH         Path to non-standard MPI installation
```

### Running Unit Tests

Running unit tests requires a working MPI installation and a
[MPI setup](https://www.open-mpi.org/faq/?category=running) between the
communicating hosts.  To install MPI, you can use standard packages provided
for your linux distribution. Once MPI is setup, you can use commands like below
for running any test of your choice.

```
mpirun -n 2 --host <host-1>,<host-2> $INSTALL_PREFIX/bin/nccl_message_transfer
```

**Note:** All tests require 2 MPI ranks to run except [ring.c](tests/ring.c)
which requires atleast 3 ranks.

### Running nccl-perf tests

To run standard `nccl-perf` tests with the `aws-ofi-nccl` plugin, you can
follow the instructions below.

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
NCCL_DEBUG=INFO build/all_reduce_perf -b 8 -f 2 -e 32M -c 1
```

If you installed the AWS libfabric plugin in a custom prefix, ensure
`LD_LIBRARY_PATH` is set to include that prefix so the perf test binaries can
find the plugin.

## Known Issues

The plugin returns only 1 NIC device even if the system supports multiple NICs.

## Getting Help

If you have any issues in building or using the package or if you think you may
have found a bug, please open an
[issue](https://github.com/aws/aws-ofi-nccl/issues).

## Contributing

Reporting issues and sending pull requests are always welcome. To learn how you
can contribute, please look at our
[contributing guidelines](CONTRIBUTING.md#contributing-guidelines).

## License

This library is licensed under the [Apache 2.0 License](LICENSE).
