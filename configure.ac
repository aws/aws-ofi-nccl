#
# Copyright (c) 2018-2019, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_PREREQ([2.69])
AC_INIT([aws-ofi-nccl], [1.4.0], [rashika@amazon.com], , [http://github.com/aws/aws-ofi-nccl])
AC_CONFIG_SRCDIR([src/nccl_ofi_net.c])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIRS([m4])
AM_SILENT_RULES([yes])
AX_CHECK_ENABLE_DEBUG([no])

# Checks for programs.
AC_PROG_CC
AC_PROG_CC_STDC
AC_PROG_MAKE_SET
AM_PROG_AR

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT64_T
AC_FUNC_MALLOC
AC_CHECK_FUNCS([memset realpath])

# Checks for libraries.
AC_ARG_WITH([libfabric],
            AC_HELP_STRING([--with-libfabric=PATH], [Path to non-standard libfabric installation]),
            [AS_IF([test -d $withval/lib64], [ofi_libdir="lib64"], [ofi_libdir="lib"])
             CFLAGS="-I$withval/include $CFLAGS"
             LDFLAGS="-L$withval/$ofi_libdir $LDFLAGS"],
            [])

AC_ARG_WITH([cuda],
            AC_HELP_STRING([--with-cuda=PATH], [Path to non-standard CUDA installation]),
            [AS_IF([test -d $withval/lib64], [cuda_libdir="lib64"], [cuda_libdir="lib"])
             CFLAGS="-I$withval/include $CFLAGS"
             LDFLAGS="-L$withval/$cuda_libdir $LDFLAGS"],
            [])

AC_ARG_WITH([nccl],
            AC_HELP_STRING([--with-nccl=PATH], [Path to non-standard NCCL installation]),
            [AS_IF([test -d $withval/lib64], [nccl_libdir="lib64"], [nccl_libdir="lib"])
             CFLAGS="-I$withval/include $CFLAGS"
             LDFLAGS="-L$withval/$nccl_libdir $LDFLAGS"],
            [])

AC_ARG_WITH([mpi],
            AC_HELP_STRING([--with-mpi=PATH], [Path to non-standard MPI installation]),
            [AS_IF([test -d $withval/lib64], [mpi_libdir="lib64"], [mpi_libdir="lib"])
             CPPFLAGS="-I$withval/include $CPPFLAGS"
             LDFLAGS="-L$withval/$mpi_libdir $LDFLAGS"],
            [])

AC_ARG_WITH([gdrcopy],
            AC_HELP_STRING([--with-gdrcopy=PATH], [Path to non-standard GDRCopy installation]),
            [AS_IF([test -d $withval/lib64], [gdrcopy_libdir="lib64"], [gdrcopy_libdir="lib"])
             CFLAGS="-I$withval/include $CFLAGS"
             LDFLAGS="-L$withval/$gdrcopy_libdir $LDFLAGS"
             AC_DEFINE([HAVE_GDRCOPY], [1], [Build support for GDRCopy])],
            [])

AM_CONDITIONAL([HAVE_GDRCOPY], [test "x$with_gdrcopy" != "x"])

AC_ARG_ENABLE([tests], [AS_HELP_STRING([--disable-tests],
      [Disable build of test binaries])], [enable_tests], [])
AM_CONDITIONAL([ENABLE_TESTS], [test "x$enable_tests" != "xno"])

# Search first for mpi.h in provided directory, then on system.
AS_IF([test "x$enable_tests" != "xno" ], AC_CHECK_HEADERS(mpi.h, [], AC_MSG_ERROR("mpi.h not found")))

AC_ARG_ENABLE(trace, [AS_HELP_STRING([--enable-trace], [Enable printing trace messages])], [], [enable_trace=no])
AC_MSG_CHECKING([whether to enable trace messages])
AS_IF([test "x${enable_trace}" = "xyes" ],
      [trace=1
       AC_MSG_RESULT(yes)],
       [trace=0
       AC_MSG_RESULT(no)])
AC_DEFINE_UNQUOTED([OFI_NCCL_TRACE], [${trace}], [Defined to 1 if aws-ofi-nccl was configured with --enable-trace, 0 otherwise])

# Checks for header files.
AC_CHECK_HEADERS([limits.h stdlib.h string.h unistd.h rdma/fabric.h cuda_runtime.h], [],[
	AC_MSG_ERROR([unable to find required headers])
])

AC_SEARCH_LIBS([dlopen], [dl], [], [
  AC_MSG_ERROR([unable to find libdl])
])
AC_SEARCH_LIBS([fi_getinfo], [fabric], [], [
  AC_MSG_ERROR([unable to find libfabric])
])

AM_INIT_AUTOMAKE([-Wall -Werror foreign subdir-objects])
AC_CONFIG_FILES([Makefile
                 src/Makefile
                 tests/Makefile])
LT_INIT([shared disable-static])
AC_OUTPUT
echo "*"
echo "* AWS OFI NCCL plugin has been configured."
echo "*"
