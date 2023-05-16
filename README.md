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
* Redhat Enterprise Linux 7 and 8
* Ubuntu 18.04 and 20.04 LTS
* CentOS 7 and 8

It requires [Libfabric](http://github.com/ofiwg/libfabric/). Please see the
[Release notes](http://github.com/aws/aws-ofi-nccl/releases) for
information on version compatibility.

Libfabric supports various providers. The plug-in can choose only those which
support the following features as defined in the
[libfabric API documentation](https://github.com/ofiwg/libfabric/tree/master/man/).

* Tagged messaging (`FI_TAGGED`, `FI_MSG`)
* Data transfer context structures (`FI_CONTEXT`)
* Reliable datagram endpoints (`FI_EP_RDM`)
* Send after Send ordering semantics (`FI_ORDER_SAS`)
* Communication with remote endpoints (`FI_REMOTE_COMM`)

For GPUDirect RDMA support, it requires these additional features from libfabric
providers. If these are not supported by any provider on system, plug-in turns off
GPUDirect RDMA support.

* Transfers to/from device memory (`FI_HMEM`)
* Remote memory operations (`FI_RMA`, `FI_READ`)

## Getting Started

### Dependencies

`aws-ofi-nccl` requires a working installation of libfabric. You can
find the instructions for installing libfabric at
[libfabric installation](https://github.com/ofiwg/libfabric).

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
  --with-mpi=PATH         Path to non-standard MPI installation
```

By default, the configure script attempts to auto-detect whether it is running
on an AWS EC2 instance, and if so enables AWS-specific optimizations. These
optimizations can be enabled regardless of build machine with the following
config option:

```
  --enable-platform-aws   Enable AWS-specific configuration and optimizations.
                          (default: Enabled if on EC2 instance)
```

To enable trace messages for debugging (disabled by default), use the
following config option:

```
   --enable-trace         Enable printing trace messages
```


LTTNG tracing is documented in the doc/tracing.md file.

To enable LTTNG tracing, use the following configuration option:

```
  --with-lttng=PATH       Path to LTTNG installation
```

By default, tests are built.  To disable building tests, use the following
config option:

```
   --disable-tests        Disable build of tests.
```

### Plugin Configurations

The plugin allows to configure the following variables at run-time according to your environment.

<table>
   <thead>
      <th>Parameter</th>
      <th>Description</th>
      <th>Type</th>
      <th>Accepted Value</th>
   </thead>
   <tr>
      <td><code>OFI_NCCL_USE_IPV6_TCP</code></td>
      <td>Allow using endpoints with IPv6 addressing format for TCP provider. Users can specify to use a preferred libfabric provider with `FI_PROVIDER` environment variable.</td>
      <td>Boolean</td>
      <td>0/1 (Default: 0)</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_TCP_EXCLUDE_IF</code></td>
      <td>List of interface names to be filtered out for TCP provider. Users can specify to use a preferred libfabric provider with `FI_PROVIDER` environment variable.</td>
      <td>String</td>
      <td>Comma-separated list of interface names (Default: "lo,docker0")</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_GDR_FLUSH_DISABLE</code></td>
      <td>Disable flush operation when using GPUDirect.</td>
      <td>Boolean</td>
      <td>0/1 (Default: 0)</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_NIC_DUP_CONNS</code></td>
      <td>Set number of NIC connections. This is used to increase hardware utilization. Applicable for P3Dn when using less number of GPUs than 8..</td>
      <td>Integer</td>
      <td>x, to set x number of connections. Only overridden for greater than 0 values (Default: 0)</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_CUDA_FLUSH_ENABLE</code></td>
      <td>When using GPUDirect use the cudaDeviceFlushGPUDirectRDMAWrites to
      enforce data consistency at the receiving GPU. Requires CUDA 11.3 or
      later. Note that this function only provides a GPU memory fence and
      requires that data has already been delivered to GPU memory. Some
      networks and PCIe configurations require an additional network-level
      flush that is not provided by this option.</td>
      <td>Boolean</td>
      <td>0/1 (Default: 0)</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_CQ_READ_COUNT</code></td>
      <td>Adjust the maximum number of completion entries that will
	  be read in a single Libfabric polling loop.  In general, users
	  should not have to adjust this value.  An array of completion
	  queue entry structures is created on the stack, so large (over
      16-32) values of this parameter may cause stack overflows.</td>
      <td>Integer</td>
      <td>Default: 4</td>
   </tr>
</table>


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
