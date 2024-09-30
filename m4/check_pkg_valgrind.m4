# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_VALGRIND], [
  valgrind_enabled=0

  AC_ARG_WITH([valgrind],
              [AS_HELP_STRING([--with-valgrind[=DIR]], [Enable address checking with valgrind])],
              [AS_IF([test "${with_valgrind}" != "no"], [valgrind_enabled=1])])

  AC_MSG_CHECKING([whether to enable valgrind])
  AS_IF([test "${valgrind_enabled}" = "1"],
        [AC_MSG_RESULT([yes])],
        [AC_MSG_RESULT([no])])

  AS_IF([test "${valgrind_enabled}" = "1" -a "${with_valgrind}" != "yes"],
        [AS_IF([test -f ${with_valgrind}/valgrind/valgrind.h],
               [CPPFLAGS="-isystem ${with_valgrind} ${CPPFLAGS}"],
               [test -f ${with_valgrind}/include/valgrind/valgrind.h],
               [CPPFLAGS="-isystem ${with_valgrind}/include ${CPPFLAGS}"],
               [AC_MSG_ERROR(valgrind.h not found in ${with_valgrind})])])

  AS_IF([test "${valgrind_enabled}" = "1"],
        [AC_CHECK_HEADERS([valgrind/valgrind.h],
                          [],
                          [AC_MSG_ERROR([valgrind.h not found])])
         # Valgrind API changed significantly with the introduction of
         # version 3.2.0. Verify that memory checking macros introduced by
         # version 3.2.0 are available.
         AC_MSG_CHECKING([for VALGRIND_MAKE_MEM_NOACCESS])
         AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
                            [[
                              #include <valgrind/memcheck.h>
                            ]],
                            [[#if !defined(VALGRIND_MAKE_MEM_NOACCESS)
                              #error Failed not defined $define
                              return 1;
                              #else
                              return 0;
                              #endif
                              ]])],
                           [AC_MSG_RESULT([yes])],
                           [AC_MSG_RESULT([no])
                            AC_MSG_ERROR([Need valgrind version 3.2.0 or later.])])])

  AC_DEFINE_UNQUOTED([ENABLE_VALGRIND], [${valgrind_enabled}], [Defined to 1 if valgrind is enabled])
])
