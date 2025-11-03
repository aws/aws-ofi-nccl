# -*- autoconf -*-
#
# Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_GDRCOPY], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([gdrcopy],
     [AS_HELP_STRING([--with-gdrcopy=PATH], [Path to non-standard gdrcopy installation])])

  AS_IF([test -n "${with_gdrcopy}"], [NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --with-gdrcopy=${with_gdrcopy}"])

  AS_IF([test -z "${with_gdrcopy}" -o "${with_gdrcopy}" = "yes"],
        [],
        [test "${with_gdrcopy}" = "no"],
        [check_pkg_found=no],
        [AS_IF([test -d ${with_gdrcopy}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
         CPPFLAGS="-isystem ${with_gdrcopy}/include ${CPPFLAGS}"
         LDFLAGS="-L${with_gdrcopy}/${check_pkg_libdir} ${LDFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([gdrapi.h], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_SEARCH_LIBS([gdr_open], [gdrapi], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_DEFINE([HAVE_GDRCOPY], [1], [Define to 1 if GDRCopy is available])
         $1],
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
