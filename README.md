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

It requires [Libfabric](http://github.com/ofiwg/libfabric/),
[NCCL](https://github.com/NVIDIA/nccl),
[HWLOC](https://www.open-mpi.org/projects/hwloc/), and (if you want to
run tests) an MPI Implementation. Please see the
[Release notes](http://github.com/aws/aws-ofi-nccl/releases) for
information on version compatibility.  We recommend using the
distribution version of hwloc, which can be installed with `yum
install hwloc-devel` on many RPM based distributions and `apt install
libhwloc-dev` on many DPKG based distibutions.

Libfabric supports various providers. The plug-in can choose only those which
support the following features as defined in the
[libfabric API documentation](https://github.com/ofiwg/libfabric/tree/master/man/).

* Tagged messaging (`FI_TAGGED`, `FI_MSG`)
* Data transfer context structures (`FI_CONTEXT`, `FI_CONTEXT2`)
* Reliable datagram endpoints (`FI_EP_RDM`)
* Send after Send ordering semantics (`FI_ORDER_SAS`)
* Communication with remote endpoints (`FI_REMOTE_COMM`)

For GPUDirect RDMA support, it requires these additional features from libfabric
providers. If these are not supported by any provider on system, plug-in turns off
GPUDirect RDMA support.

* Transfers to/from device memory (`FI_HMEM`)
* Remote memory operations (`FI_RMA`, `FI_READ`)

For multi-rail support, it requires `FI_WRITE` in addition to
`FI_READ`.

## Getting Started

### Dependencies

`aws-ofi-nccl` requires a working installation of libfabric. You can
find the instructions for installing libfabric at
[libfabric installation](https://github.com/ofiwg/libfabric).

### Build Instructions

We recommend that most users start with a release tarball available on
the
[GitHub Release Page](https://github.com/aws/aws-ofi-nccl/releases).
The plugin uses GNU autotools for its build system. You can build it
as follows:

```
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
  --with-hwloc=PATH       Path to non-standard HWLOC installation
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

To enable UBSAN (Undefined Behaviour Sanitizer), use the following config option:

```
   --enable-ubsan         Enable undefined behaviour checks with UBSAN
```

To enable memory access checks with ASAN (disabled by default), use
the following config option:

```
   --enable-asan           Enable ASAN memory access checks
```

In case plugin is configured with `--enable-asan` and the executable
binary is not compiled and linked with ASAN support, it is required to
preload the ASAN library, i.e., run the application with `export
LD_PRELOAD=<path to libasan.so>`.

In case plugin is configured with `--enable-asan` and the plugin is
run within a CUDA application, environment variable `ASAN_OPTIONS`
needs to include `protect_shadow_gap=0`. Otherwise, ASAN will crash on
out-of-memory.
NCCL currently has some memory leaks and ASAN reports memory leaks by
default on process exit. To avoid warnings on such memory leaks, e.g.,
only invalid memory accesses are of interest, add `detect_leaks=0` to
`ASAN_OPTIONS`.

To enable memory access checks with valgrind (disabled by default),
use the following config option:

```
   -with-valgrind[=PATH]  Enable valgrind memory access checks
```

Use optional parameter `PATH` to point the build to a custom path
where valgrind is installed. The memory access checkers ASAN and
valgrind are mutually exclusive.

In case plugin allocates a block of memory to store multiple
structures, redzones are added between adjacent objects such that
memory access checker can detect access out of the boundaries of these
objects. The redzones are dedicated memory areas that are marked as
not accessible by memory access checkers. The default size of redzones
is 16 bytes in case memory access checks are enabled and 0
otherwise. To control the size of redzones, use the following config
option:

```
   MEMCHECK_REDZONE_SIZE=REDZONE_SIZE   Size of redzones in bytes
```

Redzones are required to be a multiple of 8 due to ASAN shadow-map
granularity.

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

Similar to NCCL or Libfabric, the plugin dynamically loads CUDA
dependencies at runtime, specifically `libcudart.so`.  Like NCCL and
Libfabric, the plugin does not find CUDA libraries with the
`CUDA_HOME` environment variable.  `dlopen()` will use the
`LD_LIBRARY_PATH` environment variable and then your system's
default search path to find `libcudart.so`.  We do this to match NCCL
and Libfabric behaviors so that all three components find the same
CUDA installation.

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
   <tr>
      <td><code>OFI_NCCL_PROTOCOL</code></td>
      <td>Protocol to use for implementing send/recv operations.
      Default is `SENDRECV`, which uses the Libfabric tagged send/recv
      interface.  This implementation will give the best performance
      on hardware that implements tagged sends natively, and likely
      most Libfabric implementations that include an eager send
      optimization for GPU buffers.  The other valid option is `RDMA`,
      which implements a sender-managed receive queue using RDMA write
      operations and supports multi-rail channels per GPU.  The `RDMA`
      protocol is likely to work better than `SENDRECV` on networks that
      do not have an eager optimization or that have multiple NICs per
      GPU.</td>
      <td>String</td>
      <td>Default: SENDRECV</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_TOPO_FILE_WRITE_ENABLE</code></td>
      <td> When enabled and RDMA communication protocol is used, write
      NCCL topology file and set environment variable
      `NCCL_TOPO_FILE`. By default, plugin writes the NCCL topology
      file to a unique temporary file using file path template
      `/tmp/aws-ofi-nccl-topo-XXXXXX` and the file is deleted at
      normal process termination. See environment variable
      `OFI_NCCL_TOPO_FILE_TEMPLATE` to control the file
      destination.</td>
      <td>Boolean</td>
      <td>0/1 (Default: 0)</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_TOPO_FILE_TEMPLATE</code></td>
      <td>Template path to a file to control the location where NCCL
      topology is written to. In case plugin writes a NCCL topology
      file and `OFI_NCCL_TOPO_FILE_TEMPLATE` is set, plugin creates a
      unique file using the provided template and writes topology to
      that file. The last six characters of the template must be
      `XXXXXX` and will be replaced to make the filename unique. Note
      that the unique topology file will not be deleted at process
      termination in this case.</td>
      <td>String</td>
      <td>Default: Unset</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_ROUND_ROBIN_THRESHOLD</code></td>
      <td>Adjust the maximum size of `RDMA` protocol messages that are
      assigned to multi-rail channels in round-robin mode. Messages larger
      than the threshold are multiplexed over all channels to increase
      network throughput. In general, users should not have to adjust this
      value. A very small threshold may cause the `RDMA` protocol
      initialization fail since RDMA protocol control messages
      shall not be multiplexed.</td>
      <td>Integer</td>
      <td>Default: 8192</td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_NET_LATENCY</code></td>
      <td>Internode network latency in us reported to NCCL.</td>
      <td>Integer</td>
      <td>Any non-negative integer. Defaults to 0, unless the configured
      platform sets a specific value.
      </td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_EAGER_MAX_SIZE</code></td>
      <td>Eager message size limit when using RDMA protocol. Message sizes greater than
      this limit will always be sent using RDMA write instead of eagerly.</td>
      <td>Integer</td>
      <td>Any non-negative integer, though must be <= ROUND_ROBIN_THRESHOLD. Defaults to 8KiB.
      </td>
   </tr>
   <tr>
      <td><code>OFI_NCCL_DISABLE_GDR_REQUIRED_CHECK</code></td>
      <td>Disable the check for required GDR support on EC2 instances. When this check
      is disabled, the plugin can be used without GDR support even on platforms
      that support GDR (P4d and later). By default, the plugin performs the check.</td>
      <td>Boolean</td>
      <td>0/1 (Default: 0)</td>
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
make MPI=1 MPI_HOME=/path/to/mpi CUDA_HOME=/path/to/cuda NCCL_HOME=/path/to/nccl
```

3. Run perf tests
```
NCCL_DEBUG=INFO mpirun -np 2 build/all_reduce_perf -b 8 -f 2 -e 32M -c 1 -g 1
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
