#
# Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

EXTRA_DIST = \
	boost \
	efa-gda/CODE_OF_CONDUCT.md \
	efa-gda/CONTRIBUTING.md \
	efa-gda/CUDA/Makefile \
	efa-gda/CUDA/README.md \
	efa-gda/LICENSE \
	efa-gda/NOTICE \
	efa-gda/README.md \
	nccl

noinst_HEADERS = \
	efa-gda/CUDA/common/efa_cuda_dp_types.h \
	efa-gda/CUDA/device/efa_cuda_dp.cuh \
	efa-gda/CUDA/device/efa_cuda_dp_impl.cuh \
	efa-gda/CUDA/device/efa_io_defs.h \
	efa-gda/CUDA/host/efa_cuda_dp.h

if HAVE_GDAKI
AM_CPPFLAGS = \
	-I$(abs_top_srcdir)/include \
	-isystem $(abs_srcdir)/efa-gda/CUDA/common \
	-isystem $(abs_srcdir)/efa-gda/CUDA/host

noinst_LTLIBRARIES = libefa_cuda_dp.la

libefa_cuda_dp_la_SOURCES = \
	efa-gda/CUDA/host/efa_cuda_dp.cpp

# Vendored efa-dp-direct commit 81c4dc7 has a bug that causes a
# "sign-compare" error. Keep this warning exception scoped to the third-party
# helper library until the vendored source uses compatible types.
libefa_cuda_dp_la_CXXFLAGS = \
	-Wno-error=sign-compare
endif
