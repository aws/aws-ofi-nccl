# AWS OFI NCCL Release notes

# Supported Distributions
* Amazon Linux
* Amazon Linux 2
* Redhat Enterprise Linux 7.0
* Ubuntu 16.04 LTS
* CentOS 7

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
