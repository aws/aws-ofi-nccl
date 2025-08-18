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

  AC_MSG_CHECKING([if dynamically linking cuda is requested])
  AC_ARG_ENABLE([cudart-dynamic],
    [AS_HELP_STRING([--enable-cudart-dynamic],
                    [link cudart dynamically (default=link statically)])],,
    [enable_cudart_dynamic=no])
  enable_cudart_dynamic=`echo $enable_cudart_dynamic`
  case $enable_cudart_dynamic in
    yes | no) ;; # only acceptable options.
    *) AC_MSG_ERROR([unknown option '$enable_cudart_dynamic' for --enable-cudart-dynamic]) ;;
  esac
  AC_MSG_RESULT([${enable_cudart_dynamic}])
  cudart_lib="cudart"
  test "x${enable_cudart_dynamic}" = "xno" && cudart_lib="${cudart_lib}_static"

  AS_IF([test -n "${with_cuda}"], [NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --with-cuda=${with_cuda}"])

  AS_IF([test -z "${with_cuda}" -o "${with_cuda}" = "yes"],
        [],
        [test "${with_cuda}" = "no"],
        [check_pkg_found=no],
        [cuda_realpath="$(realpath ${with_cuda})"
         cuda_ldpath="${cuda_realpath}/lib64"
         CUDA_LDFLAGS="-L${cuda_ldpath}"
         CUDA_CPPFLAGS="-isystem ${cuda_realpath}/include"
         CUDA_LIBS="-l${cudart_lib} -lrt -ldl"
         LDFLAGS="${CUDA_LDFLAGS} ${LDFLAGS}"
         LIBS="${CUDA_LIBS} ${LIBS}"
         CPPFLAGS="${CUDA_CPPFLAGS} ${CPPFLAGS}"
        ])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [AC_SEARCH_LIBS(
         [cudaGetDriverEntryPoint],
         [${cudartlib}],
         [],
         [check_pkg_found=no],
         [-ldl -lrt])])

  check_cuda_gdr_flush_define=0
  AS_IF([test "${check_pkg_found}" = "yes"],
        [
        AC_MSG_CHECKING([if CUDA 11.3+ is available for GDR Write Flush support])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
        #ifndef __cplusplus
        #include <assert.h>
        #endif
        #include <cuda.h>
        static_assert(CUDA_VERSION >= 11030, "cudart>=11030 required for cuFlushGPUDirectRDMAWrites");
        ])],[ check_cuda_gdr_flush_define=1 chk_result=yes ],
            [ check_cuda_gdr_flush_define=0 chk_result=no ])
        AC_MSG_RESULT(${chk_result})
        ])

  check_cuda_dmabuf_define=0
  AS_IF([test "${check_pkg_found}" = "yes"],
        [
        AC_MSG_CHECKING([if CUDA 11.7+ is available for DMA-BUF support])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([
        #ifndef __cplusplus
        #include <assert.h>
        #endif
        #include <cuda.h>
        static_assert(CUDA_VERSION >= 11070, "cudart>=11070 required for DMABUF");
        ])],[ check_cuda_dmabuf_define=1 chk_result=yes ],
            [ check_cuda_dmabuf_define=0 chk_result=no ])
        AC_MSG_RESULT(${chk_result})
        ])

  check_cuda_dmabuf_mapping_type_pcie=0
  AS_IF([test "${check_pkg_found}" = "yes"],
            [AC_CHECK_DECL([CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE],
            [check_cuda_dmabuf_mapping_type_pcie=1],
            [check_cuda_dmabuf_mapping_type_pcie=0],
            [[#include <cuda.h>]])
      ])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [check_pkg_define=1
         $1],
        [check_pkg_define=0
         $2])

  AC_DEFINE_UNQUOTED([HAVE_CUDA], [${check_pkg_define}], [Defined to 1 if CUDA is available])
  AC_DEFINE_UNQUOTED([HAVE_CUDA_DMABUF_SUPPORT], [${check_cuda_dmabuf_define}], [Defined to 1 if CUDA DMA-BUF support is available])
  AC_DEFINE_UNQUOTED([HAVE_CUDA_DMABUF_MAPPING_TYPE_PCIE], [${check_cuda_dmabuf_mapping_type_pcie}], [Defined to 1 if CUDA DMA mapping type PCIE support is available])
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
