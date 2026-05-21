#
# Copyright (c) 2018-2026, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
AM_CPPFLAGS += $(MPI_CPPFLAGS) $(CUDA_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
LDADD = $(MPI_LIBS) $(CUDA_LIBS) $(CUDA_RUNTIME_LIBS)

# this is a little jenky, but we've always assumed we had wrapper compilers
# available for MPI.  We don't want to just override CXX to get mpicxx used,
# because we lose the C++ standard setting that happens in AX_CXX_COMPILE_STDCXX
# and Amazon Linux 2 in particular supports C++17 but defaults to C++11.
# Instead of overriding CXX, override CXXCOMPILE and CXXLINK to be the
# underlying rule, but with $(MPICXX) as the compiler and set the environment
# variable CXX so that mpicxx will pick up the -std= value.
CXXCOMPILE = OMPI_CXX="$(CXX)" MPICH_CXX="$(CXX)" \
	$(MPICXX) $(DEFS) $(DEFAULT_INCLUDES) $(INCLUDES) \
	$(AM_CPPFLAGS) $(CPPFLAGS) $(AM_CXXFLAGS) $(CXXFLAGS)

CXXLINK = OMPI_CXX="$(CXX)" MPICH_CXX="$(CXX)" \
	$(LIBTOOL) $(AM_V_lt) --tag=CXX $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=link $(MPICXX) $(AM_CXXFLAGS) \
	$(CXXFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@


if ENABLE_FUNC_TESTS
noinst_HEADERS = functional_test.h

bin_PROGRAMS = nccl_connection nccl_message_transfer ring inflight_close reuse_listen_comm gin grouped_recv gin_gdaki_misconfig_probe eager_multirecv
if ENABLE_GDAKI
if HAVE_NVCC
bin_PROGRAMS += gin_put_gdaki_gpu gin_signal_gdaki_gpu
endif
endif

base_sources = functional_test.cpp

nccl_connection_SOURCES = $(base_sources) nccl_connection.cpp
nccl_message_transfer_SOURCES = $(base_sources) nccl_message_transfer.cpp
ring_SOURCES = $(base_sources) ring.cpp
inflight_close_SOURCES = $(base_sources) inflight_close.cpp
reuse_listen_comm_SOURCES = $(base_sources) reuse_listen_comm.cpp
gin_SOURCES = $(base_sources) gin.cpp
grouped_recv_SOURCES = $(base_sources) grouped_recv.cpp
gin_gdaki_misconfig_probe_SOURCES = $(base_sources) gin_gdaki_misconfig_probe.cpp

# GDAKI GPU smoke test. .cu compiles via the .cu.o suffix rule below,
# .cpp via the existing CXXCOMPILE; the binary links via CXXLINK
# (libtool / mpicxx), same as every other test, with -lcuda added.
# nvcc -c (not -dc) fully resolves device code in the object file, so
# no nvcc-driven device link step is required. -Wno-error works around
# benign NCCL macro redefinitions that fire under -Werror.
if ENABLE_GDAKI
if HAVE_NVCC
SUFFIXES = .cu

gin_put_gdaki_gpu_SOURCES = $(base_sources) gin_put_gdaki_gpu.cu
gin_put_gdaki_gpu_LDADD = $(LDADD) -lcuda

gin_signal_gdaki_gpu_SOURCES = $(base_sources) gin_signal_gdaki_gpu.cu
gin_signal_gdaki_gpu_LDADD = $(LDADD) -lcuda

.cu.o:
	$(NVCC) -std=c++17 $(NVCC_GENCODE) \
		-ccbin=$(MPICXX) \
		-Xcompiler -fPIC \
		-Xcompiler -Wno-error \
		-I$(top_builddir)/include \
		-I$(top_srcdir)/include \
		-I$(top_srcdir)/3rd-party \
		-I$(top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include \
		-I$(top_srcdir)/3rd-party/efa-dp-direct/CUDA/src \
		$(MPI_CPPFLAGS) $(CUDA_CPPFLAGS) \
		-c -o $@ $<
endif # HAVE_NVCC
endif # ENABLE_GDAKI

endif # ENABLE_FUNC_TESTS
eager_multirecv_SOURCES = $(base_sources) eager_multirecv.cpp
