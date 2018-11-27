# AWS OFI NCCL Release notes

# Supported Distributions
* Amazon Linux
* Amazon Linux 2
* Redhat Enterprise Linux 7.0
* Ubuntu 16.04 LTS
* CentOS 7

# v0.9 release notes

First public commit as part of preview announcement

AWS OFI NCCL supports [NCCL v2.3.7+](https://github.com/NVIDIA/nccl/tree/master) and requires [libfabric v1.6.x+](https://github.com/ofiwg/libfabric/tree/master).
Please note that [current master](https://github.com/ofiwg/libfabric/commit/d32e95db02967c61eff47fc57591804769fc7dfc) of libfabric is broken for rxm providers and would require [PR-4641](https://github.com/ofiwg/libfabric/pull/4641).

The plugin has been tested with following libfabric providers:
* tcp;ofi_rxm
* sockets
* verbs;ofi_rxm
