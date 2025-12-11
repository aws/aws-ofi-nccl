/*
 * Copyright (c) 2026      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_TYPES_H
#define NCCL_OFI_GIN_TYPES_H

#include <stdint.h>

/**
 * Forward-declarations of GIN types
 */
class nccl_ofi_gin_mr_handle_t;
class nccl_ofi_gin_comm;
class nccl_ofi_gin_resources;

/**
 * Constants
 */
#define MAX_NUM_RAILS (4)

#define GIN_IMM_COMM_BITS_SIZE 20
#define GIN_MAX_COMMS (1 << GIN_IMM_COMM_BITS_SIZE)

#endif
