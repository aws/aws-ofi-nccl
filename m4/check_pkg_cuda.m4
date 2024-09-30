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
        [AS_IF([test -d $(realpath ${with_cuda})/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
         CUDA_LDFLAGS="-L$(realpath ${with_cuda})/${check_pkg_libdir}"
         CUDA_CPPFLAGS="-isystem $(realpath $(realpath ${with_cuda})/include)"
         CUDA_LIBS="-lcudart_static -lrt -ldl"
         LDFLAGS="${CUDA_LDFLAGS} ${LDFLAGS}"
         LIBS="${CUDA_LIBS} ${LIBS}"
         CPPFLAGS="${CUDA_CPPFLAGS} ${CPPFLAGS}"
        ])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_SEARCH_LIBS(
         [cudaGetDriverEntryPoint],
         [cudart_static],
         [],
         [check_pkg_found=no],
         [-ldl -lrt])])

  check_cuda_gdr_flush_define=0
  AS_IF([test "${check_pkg_found}" = "yes"],
        [
        AC_MSG_CHECKING([if CUDA 11.3+ is available for GDR Write Flush support])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
        #include <cuda.h>
        _Static_assert(CUDA_VERSION >= 11030, "cudart>=11030 required for cuFlushGPUDirectRDMAWrites");
        ])],[ check_cuda_gdr_flush_define=1 ],
            [ check_cuda_gdr_flush_define=0 ])
        AC_MSG_RESULT(${check_cuda_gdr_flush_define})
        ])

  check_cuda_dmabuf_define=0
  AS_IF([test "${check_pkg_found}" = "yes"],
        [
        AC_MSG_CHECKING([if CUDA 11.7+ is available for DMA-BUF support])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
        #include <cuda.h>
        _Static_assert(CUDA_VERSION >= 11070, "cudart>=11070 required for DMABUF");
        ])],[ check_cuda_dmabuf_define=1 ],
            [ check_cuda_dmabuf_define=0 ])
        AC_MSG_RESULT(${check_cuda_dmabuf_define})
        ])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1
         $1],
        [check_pkg_define=0
         $2])

  AC_DEFINE_UNQUOTED([HAVE_CUDA], [${check_pkg_define}], [Defined to 1 if CUDA is available])
  AC_DEFINE_UNQUOTED([HAVE_CUDA_DMABUF_SUPPORT], [${check_cuda_dmabuf_define}], [Defined to 1 if CUDA DMA-BUF support is available])
  AC_DEFINE_UNQUOTED([HAVE_CUDA_GDRFLUSH_SUPPORT], [${check_cuda_gdr_flush_define}], [Defined to 1 if CUDA cuFlushGPUDirectRDMAWrites support is available])
  AM_CONDITIONAL([HAVE_CUDA], [test "${check_pkg_found}" = "yes"])

  AC_SUBST([CUDA_LDFLAGS])
  AC_SUBST([CUDA_CPPFLAGS])
  AC_SUBST([CUDA_LIBS])

  CPPFLAGS="${check_pkg_CPPFLAGS_save}"
  LDFLAGS="${check_pkg_LDFLAGS_save}"
  LIBS="${check_pkg_LIBS_save}"

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_define])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])
