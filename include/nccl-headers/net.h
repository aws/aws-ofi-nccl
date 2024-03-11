/*
 * Copyright (c)      2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_NET_H
#define NCCL_HEADERS_NET_H

#include "config.h"

#if HAVE_CUDA
#include <cuda.h>
#include "nccl-headers/nvidia/net.h"
#elif HAVE_NEURON
#include "nccl-headers/neuron/net.h"
#else
#warning "Platform not defined"
#endif

#endif
