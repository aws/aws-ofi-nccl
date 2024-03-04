# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_LIBFABRIC], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([libfabric],
     [AS_HELP_STRING([--with-libfabric=PATH], [Path to non-standard libfabric installation])])

  AS_IF([test -n "${with_libfabric}"], [NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --with-libfabric=${with_libfabric}"])

  AS_IF([test -z "${with_libfabric}" -o "${with_libfabric}" = "yes"],
        [],
        [test "${with_libfabric}" = "no"],
        [check_pkg_found=no],
        [AS_IF([test -d ${with_libfabric}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
         CPPFLAGS="-isystem ${with_libfabric}/include ${CPPFLAGS}"
         LDFLAGS="-L${with_libfabric}/${check_pkg_libdir} ${LDFLAGS}"])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_HEADERS([rdma/fabric.h], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_SEARCH_LIBS([fi_getinfo], [fabric], [], [check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_MSG_CHECKING([for Libfabric 1.11.0 or later])
	 AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
[[#include <rdma/fabric.h>
]],
[[#if !defined(FI_MAJOR_VERSION)
#error "we cannot check the version -- sad panda"
#elif FI_VERSION_LT(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION), FI_VERSION(1,11))
#error "version is too low -- nopes"
#endif
]])],
                      [AC_MSG_RESULT([yes])],
                      [AC_MSG_RESULT([no])
                       check_pkg_found=no])])

  AS_IF([test "${check_pkg_found}" = "yes"],
	[AC_CHECK_HEADERS([rdma/fi_ext.h])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_CHECK_DECLS([FI_OPT_CUDA_API_PERMITTED,
                  FI_OPT_EFA_USE_DEVICE_RDMA,
                  FI_OPT_EFA_EMULATED_WRITE,
                  FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES,
                  FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES,
                  FI_OPT_MAX_MSG_SIZE,
                  FI_OPT_SHARED_MEMORY_PERMITTED,
                  FI_MR_DMABUF,
		  FI_PROGRESS_CONTROL_UNIFIED,
		  FI_OPT_INJECT_RMA_SIZE],
                  [], [], [AC_INCLUDES_DEFAULT
[#include <rdma/fi_endpoint.h>
#ifdef HAVE_RDMA_FI_EXT_H
#include <rdma/fi_ext.h>
#endif]])])

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
