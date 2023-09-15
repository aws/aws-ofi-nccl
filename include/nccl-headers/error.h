/*
 * Copyright (c)      2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_ERROR_H
#define NCCL_HEADERS_ERROR_H

#if HAVE_CUDA
#include "nccl-headers/nvidia/err.h"
#else
#include "nccl-headers/neuron/error.h"
#endif

#endif
