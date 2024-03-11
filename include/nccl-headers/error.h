/*
 * Copyright (c)      2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_ERROR_H
#define NCCL_HEADERS_ERROR_H

#include "config.h"

#if HAVE_CUDA
#include <cuda.h>
#include "nccl-headers/nvidia/err.h"
#elif HAVE_NEURON
#include "nccl-headers/neuron/error.h"
#else
#warning "Platform not defined"
#endif

#endif
