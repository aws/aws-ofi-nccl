# -*- Autoconf -*-
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Check for --enable-ngc flag in configure.
#
# This configure option is used only during Debian package building
# for supporting NGC images with library name libnccl-net-aws-ofi.so (aka NGC v1).
#
# With this option enabled we get another artifact by the name libnccl-net-aws-ofi.so
# which also has the same SONAME embedded in the ELF image.
# Thus, we are able to build correctly the nccl-ofi-ngc-v1 package,
# and avoid any unwanted symlinks in /opt/amazon/aws-ofi-nccl/lib/ due
# to ldconfig seeing SONAME that is different than the library name,
# as previously the target libnccl-net-ofi.so was renamed to libnccl-net-aws-ofi.so,
# during Debian package building.
#
# The same is true also for libnccl-tuner-aws-ofi.so.
#
# See contrib/debian-template/rules for sole usage of this option.
AC_DEFUN([CHECK_ENABLE_NGC],[
  AC_ARG_ENABLE([ngc],
     [AS_HELP_STRING([--enable-ngc],
         [Enable NGC container specific packaging. Creates additional plugin libraries to conform to older NGC container conventions. (default: Disabled)])])

  AM_CONDITIONAL([ENABLE_NGC], [test "${enable_ngc}" = "yes"])
  AS_IF([test "${enable_ngc}" = "yes"],
        [AC_DEFINE([ENABLE_NGC], [1], [Define to 1 if NGC container specific packaging is enabled])
         NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --enable-ngc=${enable_ngc}"])])

