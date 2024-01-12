# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_CUDA], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([cuda],
     [AS_HELP_STRING([--with-cuda=PATH], [Path to non-standard CUDA installation])])

  AS_IF([test -n "${with_cuda}"], [NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --with-cuda=${with_cuda}"])

  AS_IF([test -z "${with_cuda}" -o "${with_cuda}" = "yes"],
        [],
        [test "${with_cuda}" = "no"],
        [check_pkg_found=no],
        [AS_IF([test -d ${with_cuda}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])

         CUDA_LDFLAGS="-L${with_cuda}/${check_pkg_libdir}"

         CPPFLAGS="-I${with_cuda}/include ${CPPFLAGS}"
         LDFLAGS="${CUDA_LDFLAGS} ${LDFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([cuda.h], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_SEARCH_LIBS([cuMemHostAlloc], [cuda], [CUDA_LIBS="-lcuda"], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1
         $1],
        [check_pkg_define=0
         CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         $2])

  AC_DEFINE_UNQUOTED([HAVE_CUDA], [${check_pkg_define}], [Defined to 1 if CUDA is available])
  AM_CONDITIONAL([HAVE_CUDA], [test "${check_pkg_found}" = "yes"])

  AC_SUBST([CUDA_LDFLAGS])
  AC_SUBST([CUDA_LIBS])

  LDFLAGS="${check_pkg_LDFLAGS_save}"
  LIBS="${check_pkg_LIBS_save}"

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])
