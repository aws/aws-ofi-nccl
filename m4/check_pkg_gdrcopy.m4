# -*- autoconf -*-
#
# Copyright (c) 2025      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_GDRCOPY], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"

  AC_ARG_WITH([gdrcopy],
     [AS_HELP_STRING([--with-gdrcopy=PATH], [Path to non-standard gdrcopy installation])])

  AS_IF([test -z "${with_gdrcopy}" -o "${with_gdrcopy}" = "yes"],
        [],
        [test "${with_gdrcopy}" = "no"],
        [check_pkg_found=no],
        [CPPFLAGS="-isystem ${with_gdrcopy}/include ${CPPFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([gdrapi.h], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_DECLS([gdr_pin_buffer_v2], [], [], [AC_INCLUDES_DEFAULT
                        [#include <gdrapi.h>]])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1],
        [check_pkg_define=0])

  AC_DEFINE_UNQUOTED([HAVE_GDRCOPY], [${check_pkg_define}], [Defined to 1 if GDRCopy header is available])

  AS_IF([test "${check_pkg_found}" = "no"],
        [CPPFLAGS="${check_pkg_CPPFLAGS_save}"])

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
])
