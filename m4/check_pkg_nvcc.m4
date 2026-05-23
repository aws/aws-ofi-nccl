# -*- autoconf -*-
#
# Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# CHECK_PKG_NVCC([action-if-found], [action-if-not-found])
#
# Detect nvcc and compute a sensible NVCC_GENCODE default. Must be
# called AFTER CHECK_PKG_CUDA so that ${with_cuda}, CUDA_MAJOR, and
# CUDA_MINOR are available.
#
# Defines AM_CONDITIONAL([HAVE_NVCC]) and AC_SUBSTs NVCC, NVCC_GENCODE.
#
# NVCC_GENCODE follows NCCL's makefiles/common.mk pattern: a CUDA-
# version-conditional default targeting datacenter SKUs from sm_80
# (A100, the floor for EFA) up to whatever the toolkit supports, plus
# a PTX baseline for forward compatibility. Users can override by
# passing NVCC_GENCODE="..." on the make command line.

AC_DEFUN([CHECK_PKG_NVCC], [
  AC_ARG_WITH([nvcc],
     [AS_HELP_STRING([--with-nvcc=PATH],
        [Path to nvcc binary (defaults to ${with_cuda}/bin/nvcc, then PATH)])])

  check_nvcc_found=no

  dnl Only search for nvcc when CUDA was found. Otherwise short-circuit
  dnl so the AM_CONDITIONAL is still defined consistently.
  AS_IF([test "${have_device_interface}" != "cuda"],
        [check_nvcc_skip=yes],
        [check_nvcc_skip=no])

  AS_IF([test "${check_nvcc_skip}" = "yes"], [],
        [test -n "${with_nvcc}" -a "${with_nvcc}" != "yes" -a "${with_nvcc}" != "no"],
        [AS_IF([test -x "${with_nvcc}"],
               [NVCC="${with_nvcc}"
                check_nvcc_found=yes],
               [AC_MSG_WARN([--with-nvcc=${with_nvcc} not found or not executable])])],
        [test "${with_nvcc}" = "no"],
        [check_nvcc_found=no],
        [dnl Default search: ${with_cuda}/bin/nvcc, then PATH.
         AS_IF([test -n "${with_cuda}" -a -x "${with_cuda}/bin/nvcc"],
               [NVCC="${with_cuda}/bin/nvcc"
                check_nvcc_found=yes],
               [AC_PATH_PROG([NVCC], [nvcc], [])
                AS_IF([test -n "${NVCC}"], [check_nvcc_found=yes])])])

  AS_IF([test "${check_nvcc_found}" = "yes"],
        [AC_MSG_CHECKING([nvcc version])
         nvcc_version_string=$("${NVCC}" --version 2>/dev/null | grep "release" | head -1)
         AC_MSG_RESULT([${nvcc_version_string}])])

  dnl Compute default NVCC_GENCODE based on CUDA_MAJOR / CUDA_MINOR
  dnl exported by CHECK_PKG_CUDA. Mirrors NCCL's makefiles/common.mk
  dnl ladder, narrowed to datacenter SKUs ≥ sm_80 since EFA hosts are
  dnl A100+ in practice and CUDA 13 deprecates < sm_75 anyway.
  AS_IF([test "${check_nvcc_found}" = "yes"],
        [
        cuda12_gencode='-gencode=arch=compute_90,code=sm_90'
        cuda12_8_gencode='-gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_120,code=sm_120'
        cuda13_gencode='-gencode=arch=compute_110,code=sm_110'
        cuda12_ptx='-gencode=arch=compute_90,code=compute_90'
        cuda13_ptx='-gencode=arch=compute_120,code=compute_120'
        sm80='-gencode=arch=compute_80,code=sm_80'

        AS_IF([test "0${CUDA_MAJOR}" -ge 13],
              [NVCC_GENCODE_DEFAULT="${sm80} ${cuda12_gencode} ${cuda12_8_gencode} ${cuda13_gencode} ${cuda13_ptx}"],
              [test "0${CUDA_MAJOR}" -eq 12 -a "0${CUDA_MINOR}" -ge 8],
              [NVCC_GENCODE_DEFAULT="${sm80} ${cuda12_gencode} ${cuda12_8_gencode} ${cuda13_ptx}"],
              [test "0${CUDA_MAJOR}" -ge 11],
              [NVCC_GENCODE_DEFAULT="${sm80} ${cuda12_ptx}"],
              [NVCC_GENCODE_DEFAULT="${sm80}"])

        dnl Honor user override from environment, otherwise use default.
        AS_IF([test -z "${NVCC_GENCODE}"], [NVCC_GENCODE="${NVCC_GENCODE_DEFAULT}"])
        AC_MSG_NOTICE([NVCC_GENCODE = ${NVCC_GENCODE}])
        ])

  AS_IF([test "${check_nvcc_found}" = "yes"],
        [$1],
        [NVCC=
         NVCC_GENCODE=
         $2])

  AC_SUBST([NVCC])
  AC_SUBST([NVCC_GENCODE])
  AM_CONDITIONAL([HAVE_NVCC], [test "${check_nvcc_found}" = "yes"])

  AS_UNSET([check_nvcc_found])
  AS_UNSET([check_nvcc_skip])
  AS_UNSET([nvcc_version_string])
  AS_UNSET([cuda12_gencode])
  AS_UNSET([cuda12_8_gencode])
  AS_UNSET([cuda13_gencode])
  AS_UNSET([cuda12_ptx])
  AS_UNSET([cuda13_ptx])
  AS_UNSET([sm80])
])
