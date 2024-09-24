# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_VAR_REDZONE], [
  default_memcheck_redzone_size=16

  AC_MSG_CHECKING([for MEMCHECK_REDZONE_SIZE])
  AC_ARG_VAR(MEMCHECK_REDZONE_SIZE,
             AS_HELP_STRING([Size of added redzones (in bytes). The default size is 16 bytes in case memory access checks are enabled and 0 otherwise. Is required to be a multiple of 8. @<:@default=16@:>@]))
  AS_IF([test "${enable_asan}" != "yes" -a "${valgrind_enabled}" != "1"],[default_memcheck_redzone_size=0])
  MEMCHECK_REDZONE_SIZE=${MEMCHECK_REDZONE_SIZE:=${default_memcheck_redzone_size}}
  AC_MSG_RESULT(${MEMCHECK_REDZONE_SIZE})
  AS_IF([test "$(( MEMCHECK_REDZONE_SIZE % 8 ))" != "0"],
        [AC_MSG_ERROR([MEMCHECK_REDZONE_SIZE=${MEMCHECK_REDZONE_SIZE} is not a multiple of 8])])

  AC_DEFINE_UNQUOTED([MEMCHECK_REDZONE_SIZE], [${MEMCHECK_REDZONE_SIZE}], [Defines size of added redzones (in bytes) in case ASAN or valgrind is enabled.])

  AS_UNSET([default_memcheck_redzone_size])
])
