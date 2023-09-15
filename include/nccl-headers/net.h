/*
 * Copyright (c)      2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_HEADERS_NET_H
#define NCCL_HEADERS_NET_H

#if HAVE_CUDA
#include "nccl-headers/nvidia/net.h"
#else
#include "nccl-headers/neuron/net.h"
#endif

#endif
