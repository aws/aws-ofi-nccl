# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_MPI], [
  check_pkg_found=yes

  check_pkg_CPPFLAGS_save="${CPPFLAGS}"
  check_pkg_LDFLAGS_save="${LDFLAGS}"
  check_pkg_LIBS_save="${LIBS}"

  AC_ARG_WITH([mpi],
     [AS_HELP_STRING([--with-mpi=PATH], [Path to non-standard MPI installation])])

  AS_IF([test -n "${with_mpi}"], [NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --with-mpi=${with_mpi}"])

  AS_IF([test -z "${with_mpi}" -o "${with_mpi}" = "yes"],
        [],
        [test "${with_mpi}" = "no"],
        [check_pkg_found=no],
        [AS_IF([test -d ${with_mpi}/lib64], [check_pkg_libdir="lib64"], [check_pkg_libdir="lib"])
         mpi_bindir="${with_mpi}/bin/"

         MPI_CPPFLAGS="-isystem ${with_mpi}/include"
         MPI_LDFLAGS="-L${with_mpi}/${check_pkg_libdir}"

         CPPFLAGS="${MPI_CPPFLAGS} ${CPPFLAGS}"
         LDFLAGS="${MPI_LDFLAGS} ${LDFLAGS}"])

  MPICC=${mpi_bindir}mpicc
  MPICXX=${mpi_bindir}mpicxx

  AC_MSG_CHECKING([for working mpicc])
  ${MPICC} --help >& AS_MESSAGE_LOG_FD
  AS_IF([test $? -eq 0],
        [AC_MSG_RESULT([yes])],
        [AC_MSG_RESULT([no])
         MPICC="${CC}"
         MPICXX="${CXX}"
         AS_IF([test "${check_pkg_found}" = "yes"],
               [AC_CHECK_HEADERS([mpi.h], [], [check_pkg_found=no])])

         AS_IF([test "${check_pkg_found}" = "yes"],
               [AC_SEARCH_LIBS([MPI_Init], [mpi], [MPI_LIBS="-lmpi"], [check_pkg_found=no])])])

  AS_IF([test "${check_pkg_found}" = "yes"],
        [$1],
        [$2])

  AC_SUBST([MPICC])
  AC_SUBST([MPICXX])
  AC_SUBST([MPI_CPPFLAGS])
  AC_SUBST([MPI_LDFLAGS])
  AC_SUBST([MPI_LIBS])

  CPPFLAGS="${check_pkg_CPPFLAGS_save}"
  LDFLAGS="${check_pkg_LDFLAGS_save}"
  LIBS="${check_pkg_LIBS_save}"

  AS_UNSET([check_pkg_found])
  AS_UNSET([check_pkg_libdir])
  AS_UNSET([check_pkg_CPPFLAGS_save])
  AS_UNSET([check_pkg_LDFLAGS_save])
  AS_UNSET([check_pkg_LIBS_save])
])
