# AWS OFI NCCL Release notes

# Supported Distributions

- Amazon Linux 2
- Amazon Linux 2023
- Ubuntu 20.04 LTS, 22.04 LTS.

For releases before v1.6.0, we generally created releases from two separate
branches, an AWS-specific branch and a general release branch.  With v1.6.0, we
have unified the code into a single branch, and made the AWS-specific parts a
compile-time option.  When a feature (or entire release) only supports one of
the two variants, we note that in the release notes.

# v1.14.0-aws (2025-02-13)

This release is a general release that supports all libfabric networks.

With this release, building with platform-aws requires
[1.22.0amzn4.0](https://github.com/aws/libfabric/commits/1.22.0amzn4.0/)
or greater. AWS customers are generally recommended to track
[the latest-available EFA Installer](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-verify.html)
for performance improvements and bug fixes.

The 1.14.x release series supports
[NCCL 2.25.3-1](https://github.com/NVIDIA/nccl/releases/tag/v2.25.3-1)
while maintaining backward compatibility with older NCCL versions
([NCCL v2.17.1](https://github.com/NVIDIA/nccl/releases/tag/v2.17.1-1) and later).

Bug Fixes and Improvements:

- RDMA Transport Enhancements:
  - Added support for both FI_CONTEXT and FI_CONTEXT2
  - Fixed FI_MR_LOCAL support by properly handling registration descriptors for connection establishment messages
  - Refactored process_completions() for improved reliability and analysis tool compatibility

- SendRecv Transport Improvements:
  - Fixed FI_MR_LOCAL support with proper MR descriptors for connection establishment messages
  - Added free list support for connection establishment messages

- Tuning Improvements:
  - Modified tuner behavior to default to NCCL internal tuner on two-node configurations
  - This change addresses outlier performance issues in two-node scenarios

- Core Improvements:
  - Enhanced logging to display fabric name alongside provider name
  - Updated NCCL types.h and err.h to sync with latest ext-net example
  - Fixed size mismatch error handling

These changes improve compatibility with libfabric 2.0 and enhance the overall reliability
of the plugin, particularly in scenarios involving memory registration and connection
establishment.

