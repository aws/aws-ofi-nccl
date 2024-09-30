# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_HWLOC], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([hwloc],
     [AC_HELP_STRING([--with-hwloc=PATH], [Path to non-standard hwloc installation])])

  AS_IF([test -n "${with_hwloc}"], [NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --with-hwloc=${with_hwloc}"])

  AS_IF([test -z "${with_hwloc}" -o "${with_hwloc}" = "yes"],
        [],
        [test "${with_hwloc}" = "no"],
        [check_pkg_found=no],
        [AS_IF([test -d ${with_hwloc}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
         CPPFLAGS="-isystem ${with_hwloc}/include ${CPPFLAGS}"
         LDFLAGS="-L${with_hwloc}/${check_pkg_libdir} ${LDFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([hwloc.h], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_SEARCH_LIBS([hwloc_topology_init], [hwloc], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [$1],
        [CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         LDFLAGS="${check_pkg_LDFLAGS_save}"
         LIBS="${check_pkg_LIBS_save}"
         $2])

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])
