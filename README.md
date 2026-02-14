# AWS OFI NCCL

AWS OFI NCCL is a plug-in which enables EC2 developers to use
[libfabric](https://github.com/ofiwg/libfabric) as a network provider while
running [NVIDIA's NCCL](https://github.com/NVIDIA/nccl) based applications.

This plug-in also has support for libfabric as a network provider while
running [AMD's RCCL](https://github.com/ROCm/rccl) based applications.

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

## Getting Started

The best way to build the plugin is to start with the latest [release
package](https://github.com/aws/aws-ofi-nccl/releases).  The plugin
developers highly discourage customers from building directly from the
HEAD of a GitHub branch, as releases go through more extensive testing
than the pre-commit testing on git branches. More information about installing
the plugin from a released tarball can be found in [INSTALL.md](INSTALL.md).

Version numbers that end in `-aws` have only been tested on Amazon Web
Services Elastic Compute Cloud (EC2) instances and the Elastic Fabric
Adapter (EFA) network transport.  Customers using other networks may
experience unexpected issues with these releases, but we welcome bug
reports if that is the case.

## Basic Requirements

The plugin is regularly tested on the following operating systems:

* Amazon Linux 2 and Amazon Linux 2023
* Ubuntu 22.04 LTS and 24.04 LTS.

Other operating systems are likely to work, but are not included in our regular
regression testing. If you find an issue unique to another operating system,
GitHub issues or (better yet) patches are appreciated.

To build the plugin, you need to have
[Libfabric](http://github.com/ofiwg/libfabric/) and
[HWLOC](https://www.open-mpi.org/projects/hwloc/) installed prior to
building the plugin. If you want to run the included multi-node
tests, you also need an MPI Implementation installed.  Each release of the
plugin has a list of dependency versions in the top-level README.md
file.

The plugin does not require NCCL to be pre-installed, but obviously a
NCCL installation is required to use the plugin.  As of NCCL 2.4.8,
it is possible to use the same plugin build across multiple versions
of NCCL (such as those installed per-package with Conda-like environments).

Most Libfabric providers should work with the plugin, possibly through
a utility provider.  The plugin generally requires Reliable datagram
endpoints (`FI_EP_RDM`) with tagged messaging (`FI_TAGGED`, `FI_MSG`).
This is similar to the requirements of most MPI implementations and a
generally tested path in Libfabric.  For GPUDirect RDMA support, the
plugin also requires `FI_HMEM` support, as well as RDMA support.

## Unit Testing

The project includes a GoogleTest/GoogleMock framework for unit testing with
complete libfabric API mocking.  To build and run unit tests:

```bash
./configure --enable-gtest
make check
```

See [tests/unit/GTEST_README.md](tests/unit/GTEST_README.md) for details on
writing tests and adding new mocks.

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
