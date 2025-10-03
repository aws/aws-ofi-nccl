# -*- autoconf -*-
#
# Copyright (c) 2024      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_NVTX], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  AC_ARG_WITH([nvtx], [AS_HELP_STRING([--with-nvtx=DIR], [Enable tracing capability with NVTX @<:@default=no@:>@])])

  dnl if provided a path, add the path to CFLAGS. Otherwise assume it's provided by --with-cuda.
  AS_IF([test "x${with_nvtx}" != "xyes" -a "x${with_nvtx}" != "xno"],
        [CPPFLAGS="-isystem ${with_nvtx}/include ${CPPFLAGS}"])

  dnl Try to use CUDA's incdir flags if cuda is being used and no specific path was provided.
  AS_IF([test "x${with_nvtx}" = "xyes" -a "x${with_cuda}" != "xyes" -a "x${with_cuda}" != "xno"],
        [CPPFLAGS="${CUDA_CPPFLAGS} ${CPPFLAGS}"])

  dnl don't enable it by default.
  AS_IF([test -z "${with_nvtx}"], [check_pkg_found=no])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([nvtx3/nvToolsExt.h], [check_pkg_found=yes], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1
         $1],
        [check_pkg_define=0
         CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         $2])

  AC_DEFINE_UNQUOTED([HAVE_NVTX_TRACING], [${check_pkg_define}], [Defined to 1 if NVTX is available])

  AS_IF([test "${check_pkg_found}" = "yes" -a "x${check_pkg_define}" = "x1"],
        [nvtx_tracing=1],
        [nvtx_tracing=0])

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_CPPFLAGS_save])
])
