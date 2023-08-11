# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_ENABLE_UBSAN], [
  ubsan_flag=-fsanitize=undefined

  AC_ARG_ENABLE([ubsan],
                [AS_HELP_STRING([--enable-ubsan], [Enable undefined behaviour checks with UBSAN])])

  AC_MSG_CHECKING([whether to enable UBSAN (Undefined Behaviour Sanitizer)])
  AS_IF([test "${enable_ubsan}" = "yes"],
        [CPPFLAGS="${ubsan_flag} ${CPPFLAGS}"
         LDFLAGS="${ubsan_flag} ${LDFLAGS}"
         AC_MSG_RESULT([yes])],
        [AC_MSG_RESULT([no])])

  # Ensure that compiler and linker support UBSAN
  AS_IF([test "${enable_ubsan}" = "yes"],
        [AC_MSG_CHECKING([for support of UBSAN])
         AC_LINK_IFELSE([AC_LANG_PROGRAM(
                            [[]],
                            [[int main(void) {
                                return 0;
                              }]])],
                           [AC_MSG_RESULT([yes])],
                           [AC_MSG_RESULT([no])
                            AC_MSG_ERROR([Compiler and linker need to support flag ${ubsan_flag}])])])
  AS_UNSET([ubsan_flag])
])
