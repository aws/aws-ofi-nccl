# -*- Autoconf -*-
#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Initialization
AC_INIT([aws-ofi-nccl], [1.8.1a1-aws], [al-ofi-nccl-team@amazon.com], , [http://github.com/aws/aws-ofi-nccl])
AC_PREREQ([2.69])
AC_CONFIG_SRCDIR([src/nccl_ofi_net.c])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects tar-pax])
AM_SILENT_RULES([yes])

dnl AX_CHECK_ENABLE_DEBUG must be called before AC_PROG_CC or
dnl AM_PROG_AR
AX_CHECK_ENABLE_DEBUG([no])
dnl AM_PROG_AR must be called before LT_INIT
AM_PROG_AR

LT_INIT([shared disable-static pic-only])

NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS=

# Checks for programs
AC_PROG_CC
m4_version_prereq([2.70], [], [AC_PROG_CC_STDC])
AC_C_INLINE

dnl ac_check_enable_debug will make sure AC_PROG_CC doesn't add a
dnl bunch of debugging flags, so there's no ordering requirement
dnl between this and AC_PROG_CC.
AS_IF([test "${ax_enable_debug}" = "no"], [CFLAGS="${CFLAGS} -O3"])

CHECK_ENABLE_SANITIZER()


# Checks for header files
AC_HEADER_STDBOOL
AS_IF([test "${ac_cv_header_stdbool_h}" = "no"],
      [AC_MSG_ERROR([NCCL OFI Plugin requires stdbool.h])])

AC_CHECK_HEADER([limits.h], [], [AC_MSG_ERROR([NCCL OFI Plugin rquires limits.h])])
AC_CHECK_HEADER([stdlib.h], [], [AC_MSG_ERROR([NCCL OFI Plugin rquires stdlib.h])])
AC_CHECK_HEADER([string.h], [], [AC_MSG_ERROR([NCCL OFI Plugin rquires string.h])])
AC_CHECK_HEADER([unistd.h], [], [AC_MSG_ERROR([NCCL OFI Plugin rquires unistd.h])])


# Checks for types
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT64_T


# Checks for library functions
AC_FUNC_MALLOC
AC_CHECK_FUNC([memset], [], [AC_MSG_ERROR([NCCL OFI Plugin requires memset function.])])
AC_CHECK_FUNC([realpath], [], [AC_MSG_ERROR([NCCL OFI Plugin requires realpath function.])])
AC_SEARCH_LIBS([dlopen], [dl], [], [AC_MSG_ERROR([NCCL OFI Plugin requires dlopen])])

# Check for GCC builtin functions
CHECK_GCC_BUILTIN([__builtin_expect])
CHECK_GCC_BUILTIN([__builtin_ffsll])

# Checks for external packages
CHECK_PKG_LIBFABRIC([], [AC_MSG_ERROR([NCCL OFI Plugin could not find a working Libfabric install.])])

CHECK_PKG_LTTNG()

have_device_interface=no
CHECK_PKG_NEURON([AS_IF([test -n "${want_cuda}"],
                        [AC_MSG_ERROR([Cannot enable both CUDA and neuron.])],
                        [want_cuda=no])
                  have_device_interface=neuron]
                  AC_SUBST([OFI_NCCL_ARCHIVE], [nccom-net]))
CHECK_PKG_CUDA([have_device_interface=cuda]
               AC_SUBST([OFI_NCCL_ARCHIVE], [nccl-net]))

AS_IF([test "${have_device_interface}" = "no"],
      [AC_MSG_ERROR([NCCL OFI Plugin requires either CUDA or Neuron runtime.])])

CHECK_PKG_HWLOC([],
		[AC_MSG_ERROR([Could not find the hwloc library. Use --with-hwloc to provide the path to non-standard hwloc installation.])])

CHECK_PKG_VALGRIND()
CHECK_VAR_REDZONE()

# do we want our tests?
CHECK_PKG_MPI([found_mpi="yes"], [found_mpi="no"])

AC_ARG_ENABLE([tests],
   [AS_HELP_STRING([--disable-tests], [Disable build of functional test binaries])])
AS_IF([test "${enable_tests}" != "no" -a "${enable_neuron}" = "yes"],
      [AC_MSG_WARN([Disabling tests due to Neuron configuration.])
       enable_tests=no],
      [test "${enable_tests}" = "yes" -a "${found_mpi}" = "no"],
      [AC_MSG_ERROR([Tests requested, but MPI not found.  Aborting.])],
      [test "${found_mpi}" = "no"],
      [AC_MSG_WARN([Disabling tests due to no MPI.])
       enable_tests=no])
AM_CONDITIONAL([ENABLE_TESTS], [test "$enable_tests" != "no"])

# Enable output at the TRACE level for unit tests.
AC_ARG_ENABLE([trace],
   [AS_HELP_STRING([--enable-trace], [Enable printing trace messages in test programs])])
AC_MSG_CHECKING([whether to enable trace messages])
AS_IF([test "${enable_trace}" = "yes" ],
      [trace=1
       AC_MSG_RESULT(yes)],
      [trace=0
       AC_MSG_RESULT(no)])
AC_DEFINE_UNQUOTED([OFI_NCCL_TRACE], [${trace}], [Defined to 1 unit test output should include TRACE level])

AC_ARG_ENABLE([picky-compiler],
   [AS_HELP_STRING([--disable-picky-compiler], [Disable adding picky compiler flags.])])
AS_IF([test "${enable_picky_compiler}" != "no"],
      [picky_compiler_flags="-Wall"
       AC_MSG_NOTICE([Adding ${picky_compiler_flags} to CFLAGS.])
       CFLAGS="${CFLAGS} ${picky_compiler_flags}"
       AS_UNSET([picky_compiler_flags])])

NCCL_OFI_PLATFORM="none"
AS_IF([test "${NCCL_OFI_PLATFORM}" = "none"], [AX_CHECK_PLATFORM_AWS()])

AC_SUBST([NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS])

AS_IF([test "${valgrind_enabled}" = "1" -a "${enable_asan}" = "yes"],
      [AC_MSG_ERROR([Enabling ASAN and valgrind at the same time is not permitted])])

AC_CONFIG_FILES([Makefile
                 include/Makefile
                 src/Makefile
                 tests/Makefile
                 tests/functional/Makefile
                 tests/unit/Makefile
                 topology/Makefile])
AC_OUTPUT
echo "*"
echo "* AWS OFI NCCL plugin has been configured."
echo "*"
echo "* Platform-specific optimizations: ${NCCL_OFI_PLATFORM}"
echo "*"
