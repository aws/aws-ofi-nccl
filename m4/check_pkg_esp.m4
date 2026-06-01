# -*- autoconf -*-
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_ESP], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([esp],
     [AS_HELP_STRING([--with-esp=DIR], [Enables ESP profiling @<:@default=no@:>@])])

  AS_IF([test -z "${with_esp}" -o "${with_esp}" = "no"],
        [check_pkg_found=no],
        [test "${with_esp}" = "yes"],
        [],
        [AS_IF([test -d ${with_esp}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
         CPPFLAGS="-I${with_esp}/include ${CPPFLAGS}"
         LDFLAGS="-L${with_esp}/${check_pkg_libdir} ${LDFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_LIB([esp], [espInitialize], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1
         $1],
        [check_pkg_define=0
         CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         LDFLAGS="${check_pkg_LDFLAGS_save}"
         LIBS="${check_pkg_LIBS_save}"
         $2])

  AC_DEFINE_UNQUOTED([HAVE_LIBESP], [${check_pkg_define}], [Defined to 1 if esp is requested and available])

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])

