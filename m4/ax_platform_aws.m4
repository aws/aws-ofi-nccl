# -*- Autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#


AC_DEFUN([AX_CHECK_PLATFORM_AWS],[
  AC_ARG_ENABLE([platform-aws],
     [AS_HELP_STRING([--enable-platform-aws],
         [Enable AWS-specific configuration and optimizations.  (default: Enabled if on EC2 instance)])])

  want_platform_aws="${enable_platform_aws}"
  AS_IF([test "${want_platform_aws}" = ""],
        [AX_CHECK_EC2_INSTANCE([want_platform_aws="yes"], [want_platform_aws="no"])])

  AC_MSG_CHECKING([if want AWS platform optimizations])
  AC_MSG_RESULT([${want_platform_aws}])

  AM_CONDITIONAL([WANT_PLATFORM_AWS], [test "${want_platform_aws}" = "yes"])
  AS_IF([test "${want_platform_aws}" = "yes"], [NCCL_OFI_PLATFORM="AWS"])
])
