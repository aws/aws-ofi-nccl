# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_LTTNG], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([lttng],
     [AS_HELP_STRING([--with-lttng=DIR], [Enable tracing capability with LTTNG @<:@default=no@:>@])])

  AS_IF([test -z "${with_lttng}" -o "${with_lttng}" = "yes"],
        [],
        [test "${with_lttng}" = "no"],
        [check_pkg_found=no],
        [AS_IF([test -d ${with_lttng}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
         CPPFLAGS="-isystem ${with_lttng}/include ${CPPFLAGS}"
         LDFLAGS="-L${with_lttng}/${check_pkg_libdir} ${LDFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_LIB([lttng-ust], [lttng_ust_probe_register], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1
         $1],
        [check_pkg_define=0
         CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         LDFLAGS="${check_pkg_LDFLAGS_save}"
         LIBS="${check_pkg_LIBS_save}"
         $2])

  AC_DEFINE_UNQUOTED([HAVE_LIBLTTNG_UST], [${check_pkg_define}], [Defined to 1 if lttng-ust is requested and available])

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])
