# -*- autoconf -*-
#
# Copyright (c) 2023      Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

AC_DEFUN([CHECK_PKG_NEURON], [
  AC_ARG_ENABLE([neuron],
     [AS_HELP_STRING([--enable-neuron], [Enable Neuron pointer support])])

  AS_IF([test "${enable_neuron}" = "yes"],
        [check_pkg_define=1
         NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS="$NCCL_NET_OFI_DISTCHCK_CONFIGURE_FLAGS --enable-neuron"
         $1],
        [check_pkg_define=0
         $2])

  AC_DEFINE_UNQUOTED([HAVE_NEURON], [${check_pkg_define}], [Defined to 1 if Neuron is available])
  AM_CONDITIONAL([ENABLE_NEURON], [test "${enable_neuron}" = "yes"])
])
