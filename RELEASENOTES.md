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
