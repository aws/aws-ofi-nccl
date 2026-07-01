# -*- Autoconf -*-
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Check for --enable-histogram flag in configure.
#
# This configure option is used to enable histogram data collection and printing.
#
# With this option enabled, all OFI_HISTOGRAM_XXX() macros are enabled, and histogram
# data will be collected.
#
# See include/nccl_ofi_prof.h for the histogram API.
#
AC_DEFUN([CHECK_ENABLE_HISTOGRAM],[
  AC_ARG_ENABLE([histogram],
     [AS_HELP_STRING([--enable-histogram],
         [Enable histogram data collection and printing. (default: Disabled)])])

  AM_CONDITIONAL([ENABLE_HISTOGRAM], [test "${enable_histogram}" = "yes"])
  AS_IF([test "${enable_histogram}" = "yes"],
        [AC_DEFINE([ENABLE_HISTOGRAM], [1], [Define to 1 if histogram data collection is enabled])
         NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --enable-histogram=${enable_histogram}"])])

