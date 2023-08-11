# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_ENABLE_SANITIZER_HANDLER], [
  AS_IF([test -n "${sanitizer_flags}"],
        [CFLAGS="${sanitizer_flags} ${CFLAGS}"
         LDFLAGS="${sanitizer_flags} ${LDFLAGS}"])
])

AC_DEFUN([CHECK_ENABLE_SANITIZE_EXPAND_SANITIZER], [
  AC_ARG_ENABLE([$1], [AS_HELP_STRING([--enable-$1], [$2])])

  AC_MSG_CHECKING([whether to enable $1])
  AS_IF([test "${enable_$1}" = "yes"],
        [sanitizer_flags="${sanitizer_flags} $3"
         AC_MSG_RESULT([yes])],
        [AC_MSG_RESULT([no])])

  AS_IF([test "${enable_$1}" = "yes"],
        [AC_MSG_CHECKING([if $1 works])
         sanitizer_CFLAGS_save="${CFLAGS}"
         CFLAGS="$3 ${CFLAGS}"
         AC_LINK_IFELSE([AC_LANG_PROGRAM(
                            [[]],
                            [[int main(void) {
                                return 0;
                              }]])],
                           [AC_MSG_RESULT([yes])],
                           [AC_MSG_RESULT([no])
                            AC_MSG_ERROR([Compiler and linker need to support flag $3])])
         CFLAGS="${sanitizer_CFLAGS_save}"])
])

AC_DEFUN([CHECK_ENABLE_SANITIZER], [
  CHECK_ENABLE_SANITIZE_EXPAND_SANITIZER([ubsan],
       [Enable undefined behavior checks with UBSAN], [-fsanitize=undefined])
  AC_CONFIG_COMMANDS_PRE([CHECK_ENABLE_SANITIZER_HANDLER])
])
