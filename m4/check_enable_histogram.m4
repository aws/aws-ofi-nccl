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

  AS_IF([test "${enable_histogram}" = "yes"], [histogram_define=1], [histogram_define=0])
  AC_DEFINE_UNQUOTED([ENABLE_HISTOGRAM], [${histogram_define}], [Defined to 1 if histogram data collection is enabled, 0 otherwise])])

