# -*- autoconf -*-
#
# Copyright (c) 2024      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_GCC_BUILTIN], [
  result="no"

  AC_MSG_CHECKING([if $1 is available])
  AC_LINK_IFELSE([AC_LANG_PROGRAM([], [
    m4_case([$1],
      [__builtin_ffs], [$1(0)],
      [__builtin_ffsl], [$1(0)],
      [__builtin_ffsll], [$1(0)],
      [__builtin_expect], [$1(0, 0)]),
      [exit(1)]
      ])], [result=yes], [result=no])

  AC_MSG_RESULT([${result}])
  AS_IF([test "${result}" = "no"], [AC_MSG_ERROR([$1 not available])], [])
])
