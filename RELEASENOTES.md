# AWS OFI NCCL Release notes

# Supported Distributions

- Amazon Linux 2 and Amazon Linux 2023
- Ubuntu 22.04 LTS and 24.04 LTS

# v1.16.3 (2025-08)

The 1.16.3 release series supports [NCCL v2.27.7-1](https://github.com/NVIDIA/nccl/releases/tag/v2.27.7-1) while maintaining backward compatibility with older NCCL versions ([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

With this release, building with platform-aws requires [Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0) or greater. And it is currently tested with versions up to [Libfabric 2.1.0amzn5.0](https://github.com/aws/libfabric/commits/2.1.0amzn5.0).

### Bug Fixes and Improvements:

- Enable domain-per-thread by default on all AWS instance types for improved performance for some applications where NCCL creates multiple proxy threads

# v1.16.2 (2025-07)

The 1.16.2 release series supports [NCCL v2.27.6-1](https://github.com/NVIDIA/nccl/releases/tag/v2.27.6-1) while maintaining backward compatibility with older NCCL versions ([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

With this release, building with platform-aws requires [Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0) or greater. And it is currently tested with versions up to [Libfabric 2.1.0amzn4.0](https://github.com/aws/libfabric/commits/2.1.0amzn4.0).

### Bug Fixes and Improvements:

- Added a new platform configuration to support using the OFI NCCL plugin on the AWS p5.4xlarge instance type

# v1.16.1 (2025-07)

The 1.16.1 release series supports [NCCL v2.27.6-1](https://github.com/NVIDIA/nccl/releases/tag/v2.27.6-1) while maintaining backward compatibility with older NCCL versions (([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

With this release, building with platform-aws requires [Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0) or greater. And it is currently tested with versions up to [Libfabric 2.1.0amzn3](https://github.com/aws/libfabric/commits/2.1.0amzn3.0).

### Bug Fixes and Improvements:

- Update the PCI link speed format reported in the topology file to match kernel 5.7+
- Added SKIP_NICS_WITHOUT_ACCEL_AT_SAME_PCI_LEVEL to skip libfabric nics that do not have an accelerator at the same pci level

# v1.16.0 (2025-06)

The 1.16.0 release series supports [NCCL v2.27.5-1](https://github.com/NVIDIA/nccl/releases/tag/v2.27.5-1) while maintaining backward compatibility with older NCCL versions (([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

With this release, building with platform-aws requires [Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0) or greater. And it is currently tested with versions up to [Libfabric 2.1.0amzn3](https://github.com/aws/libfabric/commits/2.1.0amzn3.0).

### Bug Fixes and Improvements:

- On AWS platforms the following environment variables `NCCL_BUFFSIZE`, `NCCL_P2P_NET_CHUNKSIZE`, `NCCL_NVLSTREE_MAX_CHUNKSIZE`, `NCCL_NVLS_CHUNKSIZE`, `NCCL_NET_FORCE_FLUSH` may be set by the plugin
- Fix bug that prevented communicators from aborting gracefully, as part of supporting NCCL fault tolerance features
- On AWS platforms, enable collective algorithm tuner by default
- Improve P6-B200 tuner configuration to improve performance for 4 -- 32 MiB messages across node counts and large message AllReduce on 8 nodes
- Added `libnccl-tuner-ofi.so` symlink for easier configuration with `NCCL_TUNER_PLUGIN=ofi`

# v1.15.0 (2025-06)

The 1.15.x release series supports [NCCL 2.26.6-1](https://github.com/NVIDIA/nccl/releases/tag/v2.26.6-1) while maintaining backward compatibility with older NCCL versions ([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

With this release, building with platform-aws requires [Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0) or greater. And it is currently tested with versions up to [Libfabric 2.1.0amzn3](https://github.com/aws/libfabric/commits/2.1.0amzn3.0).

### Bug Fixes and Improvements:

- Build system and platform support
  - Added AWS P6-B200 platform support
  - Changed default plugin library name to `libnccl-net-ofi.so`, and by default create symlink from `libnccl-net-ofi.so` to `libnccl-net.so` to maintain backward compatibility. This allows users to set `NCCL_NET_PLUGIN=ofi` to force NCCL to use the OFI plugin for communication. Specifying `--disable-nccl-net-symlink` to configure will skip the symlink, allowing multiple plugins to be installed in the same container.
- Tuning and performance improvements
  - Added tuner support on P6-B200 for AllReduce, AllGather, and ReduceScatter regions for 0x0 and 0x7 bitmask
  - Updated default latency for P5en and P6-B200 platforms based on empirical results and analysis
- Update to use NCCL v10 API with trafficClass parameter support for future traffic prioritization
- Migrated plugin code base from C to C++
- Added support for jobs where the number of NICs per GPU is different across systems. See the `OFI_NCCL_FORCE_NUM_RAILS` runtime environment variable documentation for more information.

### OFI NCCL plugin runtime environment variable changes:

**Deprecated environment variables**

- `OFI_NCCL_RDMA_MIN_POSTED_BOUNCE_BUFFERS`
- `OFI_NCCL_RDMA_MAX_POSTED_BOUNCE_BUFFERS`

**New environment variables**

- `OFI_NCCL_SCHED_MAX_SMALL_RR_SIZE`
- `OFI_NCCL_RDMA_MIN_POSTED_EAGER_BUFFERS`
- `OFI_NCCL_RDMA_MAX_POSTED_EAGER_BUFFERS`
- `OFI_NCCL_RDMA_MIN_POSTED_CONTROL_BUFFERS`
- `OFI_NCCL_RDMA_MAX_POSTED_CONTROL_BUFFERS`
- `OFI_NCCL_CQ_SIZE`

**Updated environment variables defaults**

- `OFI_NCCL_RR_CTRL_MSG`: default changed from `0` to `1`

# v1.14.2 (2025-04)

This is a general release that is broadly applicable and is designed to be 
used with any network that can satisfy the network capabilities the plugin 
requires, as expressed through the Libfabric API's provider discovery mechanism.
We are expanding our test coverage to continue making general releases going
forward. If you would like to facilitate this effort to get coverage for
networks you intend to use the plugin with, please reach out to us.

With this release, building with platform-aws requires
[Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0)
or greater. And it is currently tested with versions
up to [Libfabric 2.1.0](https://github.com/ofiwg/libfabric/releases/tag/v2.1.0)

The 1.14.x release series supports
[NCCL 2.26.2-1](https://github.com/NVIDIA/nccl/releases/tag/v2.26.2-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

Improvements:

  - Enable DMA-BUF by default, but blocklist DMA-BUF on EFA versions 1-3 due to
    a known issue on those platforms.

# v1.14.1 (2025-04)

This is a general release that is broadly applicable and is designed to be 
used with any network that can satisfy the network capabilities the plugin 
requires, as expressed through the Libfabric API's provider discovery mechanism.
We are expanding our test coverage to continue making general releases going
forward. If you would like to facilitate this effort to get coverage for
networks you intend to use the plugin with, please reach out to us.

With this release, building with platform-aws requires
[Libfabric v1.22.0amzn4.0](https://github.com/aws/libfabric/commits/v1.22.0amzn4.0)
or greater. And it is currently tested with versions
upto [Libfabric 2.1.0](https://github.com/ofiwg/libfabric/releases/tag/v2.1.0)

The 1.14.x release series supports
[NCCL 2.26.2-1](https://github.com/NVIDIA/nccl/releases/tag/v2.26.2-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

Bug Fixes and Improvements:

  - Fixed an issue in the sendrecv protocol that would result in a false-positive
  leaking MR keys warning with some providers.

# v1.14.0 (2025-03)

Releases v1.7.0-aws through v1.13.2-aws were intended only for use on AWS P* instances.
With this release, we are resuming general releases that are broadly applicable
and is designed to be used with any network that can satisfy the network capabilities
the plugin requires, as expressed through the Libfabric API's provider discovery mechanism.
We are expanding our test coverage to continue making general releases going
forward. If you would like to facilitate this effort to get coverage for
networks you intend to use the plugin with, please reach out to us.

With this release, building with platform-aws requires
[1.22.0amzn4.0](https://github.com/aws/libfabric/commits/1.22.0amzn4.0/)
or greater. AWS customers are generally recommended to track
[the latest-available EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-verify.html)
for performance improvements and bug fixes.

The 1.14.x release series supports
[NCCL 2.26.2-1](https://github.com/NVIDIA/nccl/releases/tag/v2.26.2-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

Bug Fixes and Improvements:

- Transport Enhancements:
  - Added memory descriptor handling for control messages to properly support FI_MR_LOCAL.
  - RDMA Transport Enhancements: Added NCCL receive request early completion support when 
    provider data progress model is FI_PROGRESS_AUTO

- Tuning Improvements:
  - Modified tuner behavior to default to NCCL internal tuner on two-node configurations.
    This change addresses outlier performance issues in two-node scenarios.

These changes improve compatibility with libfabric 2.0 and enhance the overall reliability
of the plugin, particularly in scenarios involving memory registration and connection
establishment.

# v1.13.2-aws (2024-12-06)

This release is intended only for use on AWS P* instances. A general release
that supports other libfabric networks may be made in the near future.

With this release, building with platform-aws requires
[1.22.0amzn4.0](https://github.com/aws/libfabric/commits/1.22.0amzn4.0/)
or greater. AWS customers are generally recommended to track
[the latest-available EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-verify.html)
for performance improvements and bug fixes.

The 1.13.x release series supports
[NCCL 2.23.4-1](https://github.com/NVIDIA/nccl/releases/tag/v2.23.4-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

Bug Fixes:

  - Tuner Improvements:
    - Fixed algorithm selection for larger ranks and message sizes.
    - Re-calibrated the tuner for AllGather and ReduceScatter regions for 0x7 bitmask on P5en,
    optimizing performance for larger messages.
    - Added tuner support for AllGather and ReduceScatter regions for 0x0 bitmask on P5en.

  - Resolved a performance issue by preventing the eager protocol when RDMA writes are in flight,
  improving small AllReduce collective performance.

Note: dmabuf support is now turned off by default. Users can enable it explicitly using OFI_NCCL_DISABLE_DMABUF=0 if needed.

# v1.13.1-aws (2024-11-25)

This release is intended only for use on AWS P\* instances. A general release
that supports other libfabric networks may be made in the near future.

With this release, building with platform-aws requires
[1.22.0amzn4.0](https://github.com/aws/libfabric/commits/1.22.0amzn4.0/)
or greater. AWS customers are generally recommended to track
[the latest-available EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-verify.html)
for performance improvements and bug fixes.

The 1.13.x release series supports
[NCCL 2.23.4-1](https://github.com/NVIDIA/nccl/releases/tag/v2.23.4-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

This release contains no functional changes compared to v1.13.0-aws. This release merely
updates the `version` set in `AC_INIT` to include the `-aws` suffix to match the tag
name and ensure generated artifacts are named correctly.

# v1.13.0-aws (2024-11-18)

This release is intended only for use on AWS P\* instances. A general release
that supports other libfabric networks may be made in the near future.

With this release, building with platform-aws requires
[1.22.0amzn4.0](https://github.com/aws/libfabric/commits/1.22.0amzn4.0/)
or greater. AWS customers are generally recommended to track
[the latest-available EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-verify.html)
for performance improvements and bug fixes.

The 1.13.x release series supports
[NCCL 2.23.4-1](https://github.com/NVIDIA/nccl/releases/tag/v2.23.4-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

New features:

- AWS `P5en` platform support was added.

- support was added for the NCCL v3 tuner API. The tuner now supports multiple
  platforms and supports multiple collectives.

- Scheduling improvements were made to the plugin RDMA protocol. In multirail
  configurations, this is expected to balance traffic more optimally.

- dmabuf memory registration support was added. Users facing problems with
  dmabuf may disable dmabuf with `OFI_NCCL_DISABLE_DMABUF=1`.

Breaking changes:

- As mentioned above, building with support for platform-aws now requires
  libfabric version 1.22.0amzn4.0 or greater.

- Under CUDA, the plugin now statically links the CUDA runtime by default.
  Packagers preferring to dynamically link CUDA may pass
  `--enable-cudart-dynamic` at configure time to disable this.

# v1.12.1-aws

All users of v1.12.0-aws are strongly recommended to take this fix when using
EFA Installer >= 1.35.0.

Bug fixes:
* platform-aws vf sorting code produces significant performance regressions or
  crashes when used atop latest EFA driver releases. This sorting code has been
  reverted and mitigates the problem.

The plugin has been tested with following libfabric providers using tests
bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests)
suite:
* efa

# v1.12.0-aws release notes

This release is intended only for use on AWS P* instances. A general release that supports other Libfabric networks will be made in the near future. This release requires [Libfabric v1.18.0](https://github.com/ofiwg/libfabric/releases/tag/v1.18.0) or later and supports [NCCL 2.23.4-1](https://github.com/NVIDIA/nccl/releases/tag/v2.23.4-1) while maintaining backward compatibility with older NCCL versions ([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

New Features:
* Support for tuner v3 APIs
* Support for AllGather and ReduceScatter in the tuner
* Support for PAT algorithm in the tuner

Bug fixes:
* Fixed NULL pointer access in the endpoint per communicator path
* Replaced the NVLSTree option in the tuner with RING if nRanks==nNodes

The plugin has been tested with following libfabric providers using tests bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) suite:
* efa

# v1.11.0-aws release notes

This release is intended only for use on AWS P* instances. A general release that supports other Libfabric networks will be made in the near future. This release requires [Libfabric v1.18.0](https://github.com/ofiwg/libfabric/releases/tag/v1.18.0) or later and supports [NCCL 2.22.3-1](https://github.com/NVIDIA/nccl/releases/tag/v2.22.3-1) while maintaining backward compatibility with older NCCL versions ([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

New Features:
* Autogenerate topology file on P5 by default, with detected topology, instead of using a static file
* Support for AWS P5e instance type

Bug fixes:
* Fixed segfault for platform-aws builds for instance types not explicitly configured
* Fixed failure in mr cache in SENDRECV protocol for providers that don't require memory registration
* Re-enabled WRITE_IN_ORDER_ALIGNED_128_BYTES setting and check on P5.
* Added check to cause an error when using old blocking connect_v4/accept_v4 interfaces with RDMA protocol. The previous release changed connection establishment such that these interfaces cause deadlock.

The plugin has been tested with following libfabric providers using tests bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) suite:
* efa

# v1.10.0-aws release notes

This release is intended only for use on AWS P* instances. A general release that supports other Libfabric networks will be made in the near future. This release requires [Libfabric v1.18.0](https://github.com/ofiwg/libfabric/releases/tag/v1.18.0) or later and supports [NCCL 2.21.5-1](https://github.com/NVIDIA/nccl/releases/tag/v2.21.5-1) while maintaining backward compatibility with older NCCL versions ([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

New Features:
* Replaced the model-based tuner with one based on regions derived from experimental evaluations.
* Changed properties reported to NCCL to signal that registered MRs are global, in order to support user buffer registrations.
* Added the option to use different endpoints for receive communicators connected to the same source endpoint, while using a shared completion queue.
* Updated plugin to use the zero-copy path in the EFA provider for fi_send/fi_recv operations.
* Shrank the control message to 32 bytes to fit in inline data for EFA.

Bug Fixes:
* Disabled Libfabric shared memory when possible.
* Disabled RDMA eager messages on Neuron by default for better performance.
* Ensured plugin's multi-rail protocol consistently sorts rails in order of VF index for better performance.

The plugin has been tested with following libfabric providers using tests bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) suite:
* efa

# v1.9.2-aws release notes
This is a bugfix release which requires [Libfabric
v1.18.0](https://github.com/ofiwg/libfabric/releases/tag/v1.18.0) or later and
supports [NCCL 2.21.5-1](https://github.com/NVIDIA/nccl/releases/tag/v2.21.5-1)
while maintaining backward compatibility with older NCCL versions ([NCCL
v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).

Bug Fixes:
* Improved tuner model to make better decisions on P5 instances.
* Added support, in RDMA protocol, for truncation when receiving a size in the
isend call greater than the size in the correspond irecv.
* Fixed bug that prevented the tuner from getting loaded with NCCL 2.19 and
2.20.
* Fixed logging statement regarding if a domain is created per thread or per
process.
* Updated plugin to not advertise global MR support, to avoid a performance
regression in user-registered buffers.

The plugin has been tested with following libfabric providers using tests
bundled in the source code and
[nccl-tests](https://github.com/NVIDIA/nccl-tests) suite:
* efa

# v1.9.1-aws release notes
This is a bugfix release which requires [Libfabric
v1.18.0](https://github.com/ofiwg/libfabric/releases/tag/v1.18.0) or later and
supports [NCCL 2.21.5-1](https://github.com/NVIDIA/nccl/releases/tag/v2.21.5-1)
while maintaining backward compatibility with older NCCL versions ([NCCL
v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).

Bug Fixes:
* Fix release distribution generation to include missing headers introduced in
  v1.9.0. This fixes issue #382.
* Restrict libcuda link-time dependency to builds with testing enabled
* Build fixes to explicitly link against libm and libpthread used by the plugin

The plugin has been tested with following libfabric providers using tests
bundled in the source code and
[nccl-tests](https://github.com/NVIDIA/nccl-tests) suite:
* efa

# v1.9.0-aws release notes
This release requires [Libfabric v1.18.0](https://github.com/ofiwg/libfabric/releases/tag/v1.18.0)
or later and supports [NCCL 2.21.5-1](https://github.com/NVIDIA/nccl/releases/tag/v2.21.5-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).

New Features:
* Support v8 plugin interface introduced with NCCL 2.20.  This enables
  the use of the user memory registration feature recently introduced
  in NCCL.
* Update the tuner component to support v2 ext-tuner interface introduced
  with NCCL 2.21.
* Reduce ordering constraints for control messages, to reduce head of
  line blocking under congestion.

Bug Fixes:
* Increase the number of communicators to 256K (from 4K), supporting larger
  all-to-all groups.
* Improve logging in some corner case error conditions.

The plugin has been tested with following libfabric providers using tests
bundled in the source code and
[nccl-tests](https://github.com/NVIDIA/nccl-tests) suite:
* efa

# v1.8.1-aws release notes
This is a bugfix release that requires
[Libfabric v1.18.0](https://github.com/ofiwg/libfabric/releases/tag/v1.18.0) or later and
supports [NCCL v2.19.4-1](https://github.com/NVIDIA/nccl/releases/tag/v2.19.4-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).

Bug Fixes:
* Fix an issue with the ID pool's reference counting and allocation
* Improved error propagation for failed NCCL requests, allowing applications
  to fail early instead of blocking on requests that can never be completed.

The plugin has been tested with following libfabric providers using tests
bundled in the source code and
[nccl-tests](https://github.com/NVIDIA/nccl-tests) suite:
* efa

# v1.8.0-aws release notes
This release requires [Libfabric v1.18.0](https://github.com/ofiwg/libfabric/releases/tag/v1.18.0)
or later and supports [NCCL v2.19.4-1](https://github.com/NVIDIA/nccl/releases/tag/v2.19.4-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).

New Features:
* A tuner component for the plugin that picks the optimal NCCL algorithm and
  protocol at a given scale and message size.
* Improved communicator and memory region identifier management.
* Migrated from CUDA Runtime API to functional equivalents in CUDA Driver API in
  preparation for dma-buf support for memory registration. With this change, the
  plugin uses the same mechanism as NCCL to interact with the CUDA subsystem.
* No longer forcing a _flush_ operation for network operations when running with
  H100 GPUs, even when running with older NCCL versions (< v2.19.1).
* Improvements to internal device-agnostic APIs.
* Support for NCCL v7 ext-net plugin interface introduced in NCCL v2.19.3.
* Support for Ubuntu 22.04 LTS distribution.

Bug Fixes:
* Set the maximum NVLS tree chunk size used to 512KiB to recover from a
  performance regression introduced in NCCL v2.19.4, using a parameter
  introduced in NCCL v2.20.3.
* Prevent possible invocation of CUDA calls in libfabric by
  requiring a libfabric version of v1.18.0 or newer.
* Fix debug prints that reported incorrect device IDs during initialization
* Fixes to MAX_COMM computation.
* Better handling of NVLS enablement when NCCL is statically linked to
  applications
* Fixes to internal API return codes
* Configuration system fixes for Neuron builds
* Fixes to plugin environment parsing to be case insensitive
* Miscellaneous fixes that address memory leaks, NULL derefences, and compiler
  warnings.
* Updates and improvements to the project documentation.

Testing:
This release has been tested extensively with [NCCL
v2.19.4-1](https://github.com/NVIDIA/nccl/releases/tag/v2.19.4-1) for
functionality and performance. This release has also been lightly tested with
[NCCL v2.20.3-1](https://github.com/NVIDIA/nccl/releases/tag/v2.20.3-1) that was
released earlier this week. It was tested with Libfabric versions up to
[Libfabric v1.19.0](https://github.com/ofiwg/libfabric/releases/tag/v1.19.0).

The plugin has been tested with following libfabric providers using tests
bundled in the source code and
[nccl-tests](https://github.com/NVIDIA/nccl-tests) suite:
* efa

# v1.7.4-aws release notes
This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.19.3-1](https://github.com/NVIDIA/nccl/releases/tag/v2.19.3-1) while
maintaining backward compatibility with older NCCL versions ([NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).
It was tested with Libfabric versions up to
[Libfabric v1.19.0](https://github.com/ofiwg/libfabric/releases/tag/v1.19.0).

With NCCL 2.18.5 or later and v1.7.3-aws or later of the plugin,
[NVLink SHARP](https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/)
is enabled for the first time on AWS platforms.  NVLink SHARP offloads
the computation part of Allreduce collectives to the NVLink fabric,
and involves a different set of algorithms for multi-node parallelism
than previously used.  We have seen NVLink SHARP both help and hurt
performance of applications.  While NVLink SHARP is enabled by default
if NCCL 2.18.5 or later is used, users may wish to disable it by
setting `NCCL_NVLS_ENABLE=0` in the environment of your job.

New Features:
* Hard fail if GPUDirect RDMA initialization fails on an EC2 instance
  that should support GPUDirect RDMA (such as P4d.24xlarge or
  P5.48xlarge), rather than fall back to host copy buffers at
  significantly reduced performance.  Setting the environment variable
  `OFI_NCCL_DISABLE_GDR_REQUIRED_CHECK=1` will disable this behavior.
* Change the threshold at which the rdma transport switches from round
  robin to striping from 8 KiB to 256 KiB, improving the efficiency of
  large message transfers.

Bug Fixes:
* Fixed debugging output in some initialization failure cases.
* Request `FI_LOCAL_COMM` feature from Libfabric, as flush and eager
  copies are both implemented via local communication.
* Fix initialization when using the Libfabric TCP provider.
* Improve documentation on using the plugin with AWS's Elastic Fabric
  Adapter (EFA).
* Improve handling of Neuron device detection when the plugin is used
  with Tranium instances.
* Fix segfault in error case of freelist memory growth.
* The test programs that only support 2 ranks now fail with a useful
  error message if run with another number of ranks.

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) test suite:
* efa

# v1.7.3-aws release notes
This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.18.5-1](https://github.com/NVIDIA/nccl/releases/tag/v2.18.3-1) while
maintaining backward compatibility with older NCCL versions ([NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).
It was tested with Libfabric versions up to
[Libfabric v1.18.1](https://github.com/ofiwg/libfabric/releases/tag/v1.18.1).

With NCCL 2.18.5 and v1.7.3-aws of the plugin,
[NVLink SHARP](https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/)
is enabled for the first time on AWS platforms.  NVLink SHARP offloads
the computation part of Allreduce collectives to the NVLink fabric,
and involves a different set of algorithms for multi-node parallelism
than previously used.  We have seen NVLink SHARP both help and hurt
performance of applications.  While NVLink SHARP is enabled by default
if NCCL 2.18.5 or later is used, users may wish to disable it by
setting `NCCL_NVLS_ENABLE=0` in the environment of your job.

New Features:

Bug Fixes:
* Do not disable LL and LL128 protocols on P5 instances.
* Add support for g5.48xlarge instance types.
* Fix a block in use leak in the freelist implementation.
* For NCCL 2.18.5 or later, don't disable NVLS support.
* Fix bug in handling retry error issues from Libfabric in the RDMA
  transport (P5 instance types).

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) test suite:
* efa

# v1.7.2-aws release notes
This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.18.3-1](https://github.com/NVIDIA/nccl/releases/tag/v2.18.3-1) while
maintaining backward compatibility with older NCCL versions ([NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).
It was tested with Libfabric versions up to
[Libfabric v1.18.1](https://github.com/ofiwg/libfabric/releases/tag/v1.18.1).

New Features:

Bug Fixes:
* Fix compilation against CUDA versions prior to 11.3.
* Fix allocation of free lists to avoid accidently registering user
  data, which can cause corruption on fork() with older Linux kernels.
* Fix memory leak with registered bounce buffers.
* Fix improper usage of optlen in call to fi\_getopt().
* Numerous memory cleanup fixes.

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) test suite:
* efa

# v1.7.1-aws release notes
This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.18.1-1](https://github.com/NVIDIA/nccl/releases/tag/v2.18.1-1) while
maintaining backward compatibility with older NCCL versions ([NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).
It was tested with Libfabric versions up to
[Libfabric v1.18.1](https://github.com/ofiwg/libfabric/releases/tag/v1.18.1).

New Features:
* Load libcudart.so via dlopen() instead of having a linker dependency.

Bug Fixes:

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) test suite:
* efa

# v1.7.0-aws release notes
This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.18.1-1](https://github.com/NVIDIA/nccl/releases/tag/v2.18.1-1) while
maintaining backward compatibility with older NCCL versions ([NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1) and later).
It was tested with Libfabric versions up to
[Libfabric v1.18.1](https://github.com/ofiwg/libfabric/releases/tag/v1.18.1).

New Features:
* Add RDMA-write based transport with support for AWS's new P5 instance

Bug Fixes:

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) test suite:
* efa
* tcp

# v1.6.0 release notes
This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.17.1-1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).
It was tested with Libfabric versions up to
[Libfabric v1.17.1](https://github.com/ofiwg/libfabric/releases/tag/v1.17.1).

New Features:
* Add AWS platform specific code to `master` branch to support single-branch
  development and release model.
* Follow Automake conventions for Makefiles.
* Remove Travis Support as the plugin is tested using internal AWS CI
  infrastructure.

Bug Fixes:
* Avoid topology update if NCCL_TOPO_FILE is already set
* Inline allocate_stack(..) and free_stack(..) in include/stack.h
* Shortcut parameter lookup to avoid locks in fast-path.
* Free self connecting request after network transfer completes.
* Fix TCP provider on AWS p3dn by filtering the provider list before duplicating
  info entries.

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code and [nccl-tests](https://github.com/NVIDIA/nccl-tests) test suite:
* efa
* tcp; ofi_rxm

# v1.5.0 release notes

There was no general 1.5.0 release; it was limited to an AWS release.
This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.16.2](https://github.com/NVIDIA/nccl/releases/tag/v2.16.2-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).
It was tested with Libfabric versions up to
[Libfabric v1.16.1](https://github.com/ofiwg/libfabric/releases/tag/v1.16.1).

New Features:
* A single plugin build can now be used with multiple NCCL versions
  simultaneously (from NCCL v2.4.8 forward).  As a result, the
  `--with-nccl` argument is no longer necessary when building the
  plugin.
* Support for Tranium-based instance types.  Most users should
  continue to use the plugin that is included with the Neuron software
  stack, rather than building this plugin from scratch.
* Add support for flushing using CUDA's
  `cudaDeviceFlushGPUDirectRDMAWrites()` call rather than a read from
  the NIC.  We find the default read from the NIC to perform better
  for most situations.

Bug Fixes:
* Improve performance of small messages by removing redundant
  initialization of internal structures and redundant correctness
  checks throughout the codebase.
* Improve performance of applications with multiple active proxy
  threads.
* Improved pacing of Libfabric request completion polling, which will
  reduce stack memory utilization in many cases.
* Fix some compiler warnings.

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* efa

# v1.4.0 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.12.12](https://github.com/NVIDIA/nccl/releases/tag/v2.12.12-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).
It was tested with Libfabric versions up to [Libfabric v1.15.1](https://github.com/ofiwg/libfabric/releases/tag/v1.15.1).

New Features:
* Allow users to disable building the unit tests.
* Allow enable_debug flag to configure
* Fix EFA_NIC_DUP when only a single GPU is visible (AWS release only).

Bug Fixes:
* Fix compilation on CentOS 7.
* Update tag generation for control messages.
* Check for required MPI headers to build unit tests.
* Fix the active connection issue for non-blocking accepts (impacts NCCL versions 2.12 and above).
* Fix EFA_NIC_DUP when only a single GPU is visible (AWS release only).

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v1.3.0 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.12.10](https://github.com/NVIDIA/nccl/releases/tag/v2.12.10-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).
It was tested with Libfabric versions up to [Libfabric v1.14.0](https://github.com/ofiwg/libfabric/releases/tag/v1.14.0).

New Features:
* Log error-ed request entry.
* Add P4De topology (AWS release only).

Bug Fixes:
* Retry `fi_cq_readerr` until error-ed request entry is available.
* Fix crash for providers supporting multi-rail devices.
* Retry `fi_cq_readerr` until error-ed request entry is available and
  log it (AWS release only).


Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa
* psm3

# v1.2.0 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.12.7](https://github.com/NVIDIA/nccl/releases/tag/v2.12.7-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).
It was tested with Libfabric versions up to [Libfabric v1.14.0](https://github.com/ofiwg/libfabric/releases/tag/v1.14.0).

New Features:
* Add support for NCCL v2.12 with backwards compatibility to previous NCCL versions.

Bug Fixes:
* Prevent deadlock in connection establishment when using rendezvour providers.
* Enable flush operations for provider that doesn't require memory registration.
* Enable successful runs of unit-tests with flush disabled.

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa
* psm3


# v1.1.5 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.11.4](https://github.com/NVIDIA/nccl/releases/tag/v2.11.4-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).
It was tested with Libfabric versions up to [Libfabric v1.14.0](https://github.com/ofiwg/libfabric/releases/tag/v1.14.0).

New Features:
* Make use of FI_EFA_FORK_SAFE environment variable to allow Libfabric to detect when `MADV_DONTFORK`
  is not needed (#82).  This feature requires Libfabric v1.13.0 or higher.  When used with an older version
  of Libfabric, the plugin will continue to set the RDMAV_FORK_SAFE environment variable.
* Do not request FI_PROGRESS_AUTO feature when listing OFI providers; this feature is unnecessary for the plugin
  and not requesting it improves interoperability.

Bug Fixes:
* Ensure that the buffer used for flush is page aligned and allocated with `mmap` instead of `malloc`.
  This change is needed to correctly support `fork()` with `MADV_DONTFORK` (#77).
* Fix crash when used with a GDR-capable provider that does not require memory registration (#81).

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v1.1.4 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.11.4](https://github.com/NVIDIA/nccl/releases/tag/v2.11.4-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).
It was tested with Libfabric versions up to [Libfabric v1.13.2](https://github.com/ofiwg/libfabric/releases/tag/v1.13.2).

New Features:
* Print version during plugin initialization

Bug Fixes:
* Print correct error code when failing to register a memory region

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v1.1.3 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.9.9](https://github.com/NVIDIA/nccl/releases/tag/v2.9.9-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL
v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).  It was tested with Libfabric
versions up to [Libfabric v1.12.1](https://github.com/ofiwg/libfabric/releases/tag/v1.12.1).

Ubuntu 16.04 has reached end-of-life and is no longer supported starting with this release.

Bug Fixes:
* Fix bootstrap crash with NCCL 2.9.6 on P4D instances (AWS release only).

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v1.1.2 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0) and supports [NCCL v2.8.4](https://github.com/NVIDIA/nccl/releases/tag/v2.8.4-1)
while maintaining backward compatibility with older NCCL versions (upto
[NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).

It introduces the following new features and bug fixes.

New Features:
* Add support for NCCL Net v4 API

Bug Fixes:
* Handle `flush` disable configuration

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v1.1.1 release notes

There was no general 1.1.1 release; it was limited to an AWS release.
This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)and supports [NCCL v2.7.8](https://gitub.com/NVIDIA/nccl/releases/tag/v2.7.8-1)
while maintaining backward compatibility with older NCCL versions (upto
[NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).

It introduces the following new features and bug fixes.

New Features:
* Injects a static topology into NCCL for P4d hardware
* Use EFA provider supplied speed for EFA hardware.

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v1.1.0 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)and supports [NCCL v2.7.8](https://github.com/NVIDIA/nccl/releases/tag/v2.7.8-1)
while maintaining backward compatibility with older NCCL versions (upto
[NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).

It introduces the following new features and bug fixes.

New Features:
* Detect and support multi-NIC environment
* Support GPUDirect RDMA when libfabric providers support it
* Add `flush` API support for transfers using CUDA buffers

Bug Fixes:
* Enable `RDMAV_FORK_SAFE` environment variable

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v1.0.1 release notes

This release supports [NCCL v2.6.4](https://github.com/NVIDIA/nccl/releases/tag/v2.6.4-1)
while maintaining backward compatibility with older NCCL versions (upto
[NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).

It also includes bug fixes and testing enhancements.

New Features:
* Support NCCL v2.6.4
* Add validation of memory registration APIs and getProperties API in tests.

Bug Fixes:
* Use fid_mr for memory handle
* Support disabling trace messages

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v1.0.0 release notes

This release requires [Libfabric v1.9.x](https://github.com/ofiwg/libfabric/tree/v1.9.x)
and supports [NCCL v2.5.6](https://github.com/NVIDIA/nccl/releases/tag/v2.5.6-2)
It introduces changes to remove `FI_AV_TABLE` requirement from libfabric providers
and provide several bug fixes including fixing overflow issues, memory leaks and
adding completion checks for connection establishment APIs.

New Features:
* Support NCCL v2.5.6 and require Libfabric v1.9.x

Bug Fixes:
* Remove FI_AV_TABLE requirement.
* Fix missing completion check for connect API.
* Fix resource and memory leaks.

Testing:
The plugin has been tested with following libfabric providers using unit tests
bundled in the source code:
* tcp;ofi_rxm
* sockets
* efa

# v0.9.2 release notes

This release introduces changes required to support NCCLv2.4 and fixes race
condition during connection establishment by removing FI_SOURCE requirement.

New Features:
* Support NCCL provided MR register/deregister APIs.

Bug Fixes:
* Remove FI_SOURCE requirement for providers.
* Fix travis CI to build with NCCLv2.4.

Testing:
The plugin has been tested with following libfabric providers:
* tcp;ofi_rxm
* sockets
* verbs;ofi_rxm

# v0.9.1 release notes

This release makes improvements to the building and CI infrastructure. It also
includes several bug fixes. Details below:

New Features:
* Change build system to use autoconf, automake and libtool
* Add support for continuous integration using Travis CI
* Add official support for [libfabric v1.7.x](https://github.com/ofiwg/libfabric/tree/v1.7.x)

Bug Fixes:
* Remove hard-coded CUDA path when linking test binaries.
* Provide request contexts to all libfabric send/recv calls
* Readme updates and other minor fixes

Testing:
The plugin has been tested with following libfabric providers:
* tcp;ofi_rxm
* sockets
* verbs;ofi_rxm
* psm2
* efa;ofi_rxr

# v0.9 release notes

First public commit as part of preview announcement

AWS OFI NCCL supports [NCCL v2.3.7+](https://github.com/NVIDIA/nccl/tree/master) and requires [libfabric v1.6.x+](https://github.com/ofiwg/libfabric/tree/master).
Please note that [current master](https://github.com/ofiwg/libfabric/commit/d32e95db02967c61eff47fc57591804769fc7dfc) of libfabric is broken for rxm providers and would require [PR-4641](https://github.com/ofiwg/libfabric/pull/4641).

The plugin has been tested with following libfabric providers:
* tcp;ofi_rxm
* sockets
* verbs;ofi_rxm

