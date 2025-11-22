# -*- Autoconf -*-
#
# Copyright (c) 2018-2023, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Initialization
AC_INIT([aws-ofi-nccl], [GitHub-dev], [al-ofi-nccl-team@amazon.com], , [http://github.com/aws/aws-ofi-nccl])
AC_PREREQ([2.69])
AC_CONFIG_SRCDIR([src/nccl_ofi_net.cpp])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADERS([include/config.h])
AC_CONFIG_MACRO_DIRS([m4])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects tar-pax])
AM_SILENT_RULES([yes])

AH_TOP([/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_CONFIG_H
#define NCCL_OFI_CONFIG_H

])
AH_BOTTOM([
#include "nccl_ofi_config_bottom.h"
#endif /* NCCL_OFI_CONFIG_H */
])

dnl AX_CHECK_ENABLE_DEBUG must be called before AC_PROG_CC or
dnl AM_PROG_AR
AX_CHECK_ENABLE_DEBUG([no])

# Checks for programs

dnl Enable _GNU_SOURCE and other system extensions. The plugin uses several
dnl nonstandard system functions, and the plugin only supports Linux OSes, so
dnl the risk of silently enabling system extensions is minimal.
AC_USE_SYSTEM_EXTENSIONS

AC_PROG_CXX
AC_LANG([C++])

dnl use C++17, disable runtime type information
AX_CXX_COMPILE_STDCXX([17], [noext], [mandatory])
CXXFLAGS="${CXXFLAGS} -fno-rtti"

dnl AM_PROG_AR must be called before LT_INIT
AM_PROG_AR
AC_PROG_RANLIB
LT_INIT([shared disable-static pic-only])

NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS=

dnl ac_check_enable_debug will make sure AC_PROG_CC doesn't add a
dnl bunch of debugging flags, so there's no ordering requirement
dnl between this and AC_PROG_CC.
AS_IF([test "${ax_enable_debug}" = "no"], [
             dnl Enable O3 optimization
             CXXFLAGS="${CXXFLAGS} -O3"

             dnl dead code elim
             CXXFLAGS="${CXXFLAGS} -ffunction-sections -fdata-sections"

             dnl https://maskray.me/blog/2021-05-09-fno-semantic-interposition
             CXXFLAGS="${CXXFLAGS} -fno-semantic-interposition -fvisibility=hidden "
             LDFLAGS="${LDFLAGS} -Bsymbolic"
            ])
CHECK_ENABLE_SANITIZER()


# Checks for header files
AC_CHECK_HEADER([limits.h], [], [AC_MSG_ERROR([NCCL OFI Plugin requires limits.h])])
AC_CHECK_HEADER([stdlib.h], [], [AC_MSG_ERROR([NCCL OFI Plugin requires stdlib.h])])
AC_CHECK_HEADER([string.h], [], [AC_MSG_ERROR([NCCL OFI Plugin requires string.h])])
AC_CHECK_HEADER([unistd.h], [], [AC_MSG_ERROR([NCCL OFI Plugin requires unistd.h])])

AC_CHECK_HEADERS([linux/limits.h])

# Checks for types
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT64_T


# Checks for library functions
AC_FUNC_MALLOC
AC_CHECK_FUNC([memset], [], [AC_MSG_ERROR([NCCL OFI Plugin requires memset function.])])
AC_CHECK_FUNC([realpath], [], [AC_MSG_ERROR([NCCL OFI Plugin requires realpath function.])])
AC_CHECK_FUNCS([gettid])
AC_SEARCH_LIBS([dlopen], [dl], [], [AC_MSG_ERROR([NCCL OFI Plugin requires dlopen])])

dnl it is insufficient to just link against pthread with -lpthread
dnl background: https://stackoverflow.com/a/23251828
CPPFLAGS="${CPPFLAGS} -pthread"
LDFLAGS="${LDFLAGS} -pthread"
AC_SEARCH_LIBS([pthread_mutexattr_settype], [pthread], [], [AC_MSG_ERROR([NCCL OFI Plugin requires pthreads.])])

AC_SEARCH_LIBS([log2], [m], [], [AC_MSG_ERROR([NCCL OFI Plugin requires the log2 library function.])])

dnl Need at least glibc 2.3 or later (released 2002-10-02) , because
dnl 2.2.3 added support for atexit() in shared libraries.
AC_MSG_CHECKING([for glibc 2.3 or later])
AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
  [[#include <gnu/libc-version.h>
  ]],
  [[#if __GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 3)
#error "Glibc too old to support atexit."
#endif
  ]])],
    [AC_MSG_RESULT([yes])],
    [AC_MSG_RESULT([no])
     AC_MSG_ERROR([Found glibc prior to version 2.3.  Version 2.3 or later required])])

# Checks for external packages
CHECK_PKG_LIBFABRIC([], [AC_MSG_ERROR([NCCL OFI Plugin could not find a working Libfabric install.])])

have_device_interface=no
CHECK_PKG_NEURON([AS_IF([test -n "${want_cuda}"],
                        [AC_MSG_ERROR([Cannot enable both CUDA and neuron.])],
                        [want_cuda=no])
                  have_device_interface=neuron])


CHECK_PKG_CUDA([
  AS_IF([test "${have_device_interface}" = "no" -a -z "$with_rocm"], [
    have_device_interface=cuda
  ])
])


CHECK_PKG_ROCM([
  AS_IF([test "${have_device_interface}" = "no"], [
    have_device_interface=rocm
  ])
])


AS_IF([test "${have_device_interface}" = "no"],
      [AC_MSG_ERROR([NCCL OFI Plugin requires either CUDA, ROCm, or Neuron runtime.])])

CHECK_PKG_LTTNG()
CHECK_PKG_NVTX()

do_cuda=0
do_rocm=0
AS_IF([test "$have_device_interface" = "cuda"], [do_cuda=1])
AS_IF([test "$have_device_interface" = "rocm"], [do_rocm=1; enable_tests="no"])

CHECK_PKG_HWLOC([],
		[AC_MSG_ERROR([Could not find the hwloc library. Use --with-hwloc to provide the path to non-standard hwloc installation.])])
AC_DEFINE_UNQUOTED([HAVE_CUDA], [${do_cuda}], [Defined to 1 if CUDA is available])
AM_CONDITIONAL([HAVE_CUDA], [test ${do_cuda} = 1])

CHECK_PKG_GDRCOPY()

CHECK_PKG_VALGRIND()
CHECK_VAR_REDZONE()

AC_DEFINE_UNQUOTED([HAVE_ROCM], [${do_rocm}], [Defined to 1 if ROCm is available])
AM_CONDITIONAL([HAVE_ROCM], [test ${do_rocm} = 1])

AM_CONDITIONAL([ENABLE_NEURON], [test "x${have_device_interface}" = "xneuron"])
AS_IF([test ${do_rocm} = 1],
  [AC_DEFINE_UNQUOTED([__HIP_PLATFORM_AMD__], [1], [Select AMD/ROCm HIP APIs])])

AS_CASE([$have_device_interface],
  [neuron], [DEVICE_INTERFACE=neuron],
  [*],      [DEVICE_INTERFACE=cuda])
AC_SUBST([DEVICE_INTERFACE])

NCCL_OFI_PLATFORM="none"
AS_IF([test "${NCCL_OFI_PLATFORM}" = "none"], [AX_CHECK_PLATFORM_AWS()])

AS_IF([test "${valgrind_enabled}" = "1" -a "${enable_asan}" = "yes"],
      [AC_MSG_ERROR([Enabling ASAN and valgrind at the same time is not permitted])])

CHECK_ENABLE_MEMFD_CREATE()

# do we want our tests?
CHECK_PKG_MPI([found_mpi="yes"], [found_mpi="no"])

AC_ARG_ENABLE([tests],
   [AS_HELP_STRING([--disable-tests], [Disable build of unit/functional test binaries])])

build_unit_tests=yes
build_func_tests=yes
AS_IF([test "${enable_tests}" == "no"],
      [build_unit_tests=no build_func_tests=no])
AS_IF([test "${enable_tests}" != "no" -a "${found_mpi}" = "no" ],
      [AC_MSG_WARN([Cannot build functional tests without MPI. Disabling functional tests.])
       build_func_tests=no])
AS_IF([test "${enable_tests}" != "no" -a "${have_device_interface}" = "neuron" ],
      [AC_MSG_WARN([Functional tests not available on neuron, disabling.])
       build_func_tests=no])
AM_CONDITIONAL([ENABLE_UNIT_TESTS], [test "$build_unit_tests" == "yes"])
AM_CONDITIONAL([ENABLE_FUNC_TESTS], [test "$build_func_tests" == "yes"])

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

picky_cxxflags=""
AC_DEFUN([ADD_PICKY_FLAGS],[
    AC_LANG_PUSH([C++])
    AX_CHECK_COMPILE_FLAG([$1], [picky_cxxflags="${picky_cxxflags} $1"], [], [-Werror])
    AC_LANG_POP()
    ])

AC_ARG_ENABLE([picky-compiler],
   [AS_HELP_STRING([--disable-picky-compiler], [Disable adding picky compiler flags.])])
AS_IF([test "${enable_picky_compiler}" != "no"], [
    dnl standard and normal
    ADD_PICKY_FLAGS([-Wall -Wextra])

    dnl some top-level API functions have unused parameters, but we have to keep
    dnl them. TODO: explicitly annotate those functions, rather than setting this
    dnl globally.
    ADD_PICKY_FLAGS([-Wno-unused-parameter])

    dnl warn when a variable shadows another variable.
    ADD_PICKY_FLAGS([-Wshadow])
    dnl warn on security issues around functions that format output (ie printf)
    ADD_PICKY_FLAGS([-Wformat=2])
    dnl warn for potential performance problem casts
    ADD_PICKY_FLAGS([-Wcast-align])
    dnl warn if a null dereference is detected
    ADD_PICKY_FLAGS([-Wnull-dereference])

    dnl warn when a case statement is missing a break or a line that contains a
    dnl comment before the next case.
    ADD_PICKY_FLAGS([-Wimplicit-fallthrough=2])

    dnl functions should either have internal linkage (static) or there should be a
    dnl header declarating the function.
    ADD_PICKY_FLAGS([-Wmissing-declarations])

    dnl try to enable GCC's static analyzer.  We've tripped over a number of
    dnl problems with GNU's static analyzer and C++ code, especially regarding
    dnl NULL return and leak reports from simple inheritance (as one example:
    dnl https://gcc.gnu.org/bugzilla/show_bug.cgi?id=94355).  We really loved
    dnl the analyzer for C code and would like it back, but even the simple
    dnl example below trips multiple issues which really prevent the tool from
    dnl being useful (as of GCC 14.2.0).  So we disable it in hopes that one day
    dnl we can re-enable it.
    dnl
    dnl    class A {
    dnl    public:
    dnl        A(int num) : vec(num) { }
    dnl        A() { }
    dnl        std::vector<int> vec;
    dnl    };
    dnl
    dnl    class B : public A {
    dnl    public:
    dnl        B(int num) : A(num) { }
    dnl    };
    dnl
    dnl    auto b = new B(10);
    dnl    delete b;
    dnl
    dnl ADD_PICKY_FLAGS([-fanalyzer])

    dnl todo: clang-sa flags

    AC_MSG_NOTICE([Adding ${picky_cxxflags} to CXXFLAGS.])
    CXXFLAGS="${CXXFLAGS} ${picky_cxxflags}"
])
AS_UNSET([picky_cxxflags])

werror_flags="-Werror"
AC_ARG_ENABLE([werror],
   [AS_HELP_STRING([--enable-werror], [(Developer) Enable setting -Werror.  Off by default, unless building from Git tree.])])
AS_IF([test -d "${srcdir}/.git" -a -z "${enable_werror}"],
      [AC_MSG_NOTICE([Found .git directory.  Adding ${werror_flags} to CXXFLAGS.])
       CXXFLAGS="${werror_flags} ${CXXFLAGS} "],
      [test "${enable_werror}" = "yes"],
      [AC_MSG_NOTICE([Adding ${werror_flags} to CXXFLAGS.])
       CXXFLAGS="${werror_flags} ${CXXFLAGS} "])

AC_ARG_ENABLE([nccl-net-library],
      [AS_HELP_STRING([--disable-nccl-net-library],
           [Do not create shared library libnccl-net.so that is the same as libnccl-net-ofi.so.
            Useful in cases where you are trying to install multiple plugins in the same container.])])
AC_ARG_ENABLE([nccl-net-symlink],
      [AS_HELP_STRING([--disable-nccl-net-symlink],
           [Deprecated, is equivalent to --disable-nccl-net-library. Do not create
            shared library libnccl-net.so that is the same as libnccl-net-ofi.so.
            Useful in cases where you are trying to install multiple plugins in the same container.])])
AM_CONDITIONAL([ENABLE_NCCL_NET_LIBRARY], [test "${enable_nccl_net_library}" != "no" && test "${enable_nccl_net_symlink}" != "no"])

AC_SUBST([NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS])

AC_CONFIG_FILES([Makefile
                 include/Makefile
                 3rd-party/Makefile
                 src/Makefile
                 tests/Makefile
                 tests/functional/Makefile
                 tests/unit/Makefile
                 topology/Makefile])
AC_OUTPUT
echo "*"
echo "* AWS OFI NCCL plugin has been configured."
echo "*"
echo "* Enabled Platforms: ${ENABLED_PLATFORMS}"
echo "*"
