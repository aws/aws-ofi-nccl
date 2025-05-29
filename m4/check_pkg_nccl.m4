# -*- autoconf -*-
#
# Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_NCCL], [
      check_pkg_found=yes
      check_pkg_CPPFLAGS_save="${CPPFLAGS}"
      check_pkg_LDFLAGS_save="${LDFLAGS}"
      check_pkg_LIBS_save="${LIBS}"

      AC_ARG_WITH([nccl],
            [AS_HELP_STRING([--with-nccl=PATH], [Path to non-standard NCCL installation])])

      AS_IF([test -n "${with_nccl}"], 
            [NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --with-nccl=${with_nccl}"])

      AS_IF([test -z "${with_nccl}" -o "${with_nccl}" = "yes"],
            [],
            [test "${with_nccl}" = "no"],
            [check_pkg_found=no],
            [AS_IF([test -d ${with_nccl}/lib64], 
                  [check_pkg_libdir="lib64"], 
                  [check_pkg_libdir="lib"])
            NCCL_CPPFLAGS="-isystem ${with_nccl}/include"
            CPPFLAGS="${NCCL_CPPFLAGS} ${CUDA_CPPFLAGS} ${CPPFLAGS}"
            NCCL_LDFLAGS="-L${with_nccl}/${check_pkg_libdir}"
            LDFLAGS="${NCCL_LDFLAGS} ${LDFLAGS}"
            NCCL_LIBS="-lnccl"
            LIBS="${NCCL_LIBS} ${LIBS}"])

      AS_IF([test "${check_pkg_found}" = "yes"],
            [AC_CHECK_HEADERS([nccl.h], 
            [],
            [check_pkg_found=no])])

      AS_IF([test "${check_pkg_found}" = "yes"],
            [$1], # found_nccl="yes"
            [$2]) # found_nccl="no"

      AC_SUBST([NCCL_CPPFLAGS])
      AC_SUBST([NCCL_LDFLAGS])
      AC_SUBST([NCCL_LIBS])

      CPPFLAGS="${check_pkg_CPPFLAGS_save}"
      LDFLAGS="${check_pkg_LDFLAGS_save}"
      LIBS="${check_pkg_LIBS_save}"

      AS_UNSET([check_pkg_found])
      AS_UNSET([check_pkg_libdir])
      AS_UNSET([check_pkg_CPPFLAGS_save])
      AS_UNSET([check_pkg_LDFLAGS_save])
      AS_UNSET([check_pkg_LIBS_save])
])
