# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_CUDA], [
  check_pkg_found=yes
  check_pkg_CPPFLAGS_save="${CPPFLAGS}"

  AC_ARG_WITH([cuda],
     [AS_HELP_STRING([--with-cuda=PATH], [Path to non-standard CUDA installation])])

  AS_IF([test -n "${with_cuda}"], [NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --with-cuda=${with_cuda}"])

  AS_IF([test -z "${with_cuda}" -o "${with_cuda}" = "yes"],
        [],
        [test "${with_cuda}" = "no"],
        [check_pkg_found=no],
        [CPPFLAGS="-I${with_cuda}/include ${CPPFLAGS}"]
        )

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([cuda.h], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1
         $1],
        [check_pkg_define=0
         CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         $2])

  AC_DEFINE_UNQUOTED([HAVE_CUDA], [${check_pkg_define}], [Defined to 1 if CUDA is available])
  AM_CONDITIONAL([HAVE_CUDA], [test "${check_pkg_found}" = "yes"])

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])
