# AWS OFI NCCL Release notes

# Supported Distributions
* Amazon Linux
* Amazon Linux 2
* Redhat Enterprise Linux 7.0 and 8.0
* Ubuntu 18.04 and 20.04 LTS
* CentOS 7 and 8

# v1.4.0 release notes

This release requires [Libfabric v1.11.0](https://github.com/ofiwg/libfabric/releases/tag/v1.11.0)
or later and supports [NCCL v2.12.12](https://github.com/NVIDIA/nccl/releases/tag/v2.12.12-1) while
maintaining backward compatibility with older NCCL versions (up to [NCCL v2.4.8](https://github.com/NVIDIA/nccl/releases/tag/v2.4.8-1)).
It was tested with Libfabric versions up to [Libfabric v1.15.1](https://github.com/ofiwg/libfabric/releases/tag/v1.15.1).

New Features:
* Allow users to disable building the unit tests.
* Allow enable_debug flag to configure

Bug Fixes:
* Fix compilation on CentOS 7.
* Update tag generation for control messages.
* Check for required MPI headers to build unit tests.
* Fix the active connection issue for non-blocking accepts (impacts NCCL versions 2.12 and above).

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

Bug Fixes:
* Retry `fi_cq_readerr` until error-ed request entry is available.
* Fix crash for providers supporting multi-rail devices.

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
