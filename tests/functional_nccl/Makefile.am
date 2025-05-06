#
# Copyright (c) 2025, Amazon.com, Inc. or its affiliates. All rights reserved.
#

# Please remember to update .gitignore in this directory.

CC = $(MPICC)
CXX = $(MPICXX)

if ENABLE_NCCL_FUNC_TESTS

AM_CPPFLAGS = $(MPI_CPPFLAGS) $(CUDA_CPPFLAGS) $(NCCL_CPPFLAGS)
AM_LDFLAGS = $(MPI_LDFLAGS) $(CUDA_LDFLAGS) $(NCCL_LDFLAGS)
LDADD = $(MPI_LIBS) $(CUDA_LIBS) $(NCCL_LIBS)

endif
