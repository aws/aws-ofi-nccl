# -*- autoconf -*-
#
# Copyright (c) 2024      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_ENABLE_MEMFD_CREATE], [
      AC_CHECK_FUNCS(memfd_create)
      if test "x$ac_cv_func_memfd_create" != xyes; then
            # check if we can use workaround syscall
            AC_CHECK_DECLS(SYS_memfd_create, [], [], [#include <sys/syscall.h>])
            if test "x$ac_cv_have_decl_SYS_memfd_create" != xyes; then
                  AC_MSG_ERROR("Could not find memfd_create or SYS_memfd_create")
            fi
      fi
])
