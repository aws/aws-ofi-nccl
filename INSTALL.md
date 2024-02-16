### Build Instructions

We strongly recommend starting with a release tarball available on the
[GitHub Release Page](https://github.com/aws/aws-ofi-nccl/releases) when
building from source for production uses.

`aws-ofi-nccl` requires a working installation of Libfabric (v1.18.0 or newer). You can
find the instructions for installing libfabric at
[libfabric installation](https://github.com/ofiwg/libfabric).

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
