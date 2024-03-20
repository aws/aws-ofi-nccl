# -*- autoconf -*-
#
# Copyright (c) 2024      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_NVTX], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([nvtx],
     [AS_HELP_STRING([--with-nvtx=DIR], [Enable tracing capability with NVTX @<:@default=no@:>@])])

  AS_IF([test -z "${with_nvtx}" -o "${with_nvtx}" = "yes"],
        [],
        [test "${with_nvtx}" = "no"],
        [check_pkg_found=no],
        [AS_IF([test -d ${with_nvtx}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
         CPPFLAGS="-I${with_nvtx}/include ${CPPFLAGS}"
         LDFLAGS="-L${with_nvtx}/${check_pkg_libdir} ${LDFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_LIB([nvToolsExt], [nvtxRangePop], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1
         $1],
        [check_pkg_define=0
         CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         LDFLAGS="${check_pkg_LDFLAGS_save}"
         LIBS="${check_pkg_LIBS_save}"
         $2])

  AC_DEFINE_UNQUOTED([HAVE_NVTX_TRACING], [${check_pkg_define}], [Defined to 1 if NVTX is available])

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])
