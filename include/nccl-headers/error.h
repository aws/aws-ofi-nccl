/*
 * Copyright (c) 2023-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_ERROR_H
#define NCCL_HEADERS_ERROR_H

#if HAVE_CUDA || HAVE_ROCM
#include "nccl-headers/nvidia/err.h"
#elif HAVE_NEURON
#include "nccl-headers/neuron/error.h"
#else
#error "Neither CUDA nor Neuron support is available"
#endif

#endif
