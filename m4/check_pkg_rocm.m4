# -*- autoconf -*-
#
# Copyright (c) 2024      Hewlett Packard Enterprise Development LP
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_ROCM], [
  check_pkg_found="yes"
  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([rocm],
             [AS_HELP_STRING([--with-rocm=PATH], [Path to non-standard ROCm installation])])

  AS_IF([test -z "${with-rocm}" -o "{with_rocm}" = "yes"],
        [],
        [test "${with_rocm}" = "no"],
        [check_pkg_found=no],
        [AS_IF([test -d ${with_rocm}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
        CPPFLAGS="-I${with_rocm}/include ${CPPFLAGS}"
        LDFLAGS="-L${with_rocm}/${check_pkg_libdir} ${LDFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_LIB([amdhip64], [hipMemAllocHost], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([hip/hip_runtime_api.h], [], [check_pkg_found=no], [#define __HIP_PLATFORM_AMD__])])


  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define="yes"],
        [check_pkg_define="no"
         CPPFLAGS="${check_pkg_CPPFLAGS_save}"
         LDFLAGS="${check_pkg_LDFLAGS_save}"
         LIBS="${check_pkg_LIBS_save}"
        ])

  AS_IF([test -n "${with_rocm}"],
       [AS_IF([test "${check_pkg_define}" = "yes"],
              [$1], [$2] )
       ], [$2]
   )

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])
