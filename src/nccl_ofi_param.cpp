/*
 * Copyright (c) 2024      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

/*
 * This is an ugly hack.  The original implementation of
 * nccl_ofi_param created inline functions to access each environment
 * variable, using the macros found in nccl_ofi_param.h.  However,
 * this creates something of an ODR problem, as multiple complication
 * units can call the same param lookup function, and that results in
 * naming conflicts.  So instead, we have the header file act like a
 * normal header file most of the time, and when included from
 * nccl_ofi_param.c with OFI_NCCL_PARAM_DEFINE set to 1, stamps out
 * the original implementations of the functions.  So now we have one
 * copy of each function that everyone can call.
 *
 * This is intended to be a transient state.  We want to rewrite the
 * entire param system once we've finished moving to C++, but need to
 * solve the ODR problem before we move to C++.  So here lies one of
 * the more terrible pieces of code I've ever written.
 */
#define OFI_NCCL_PARAM_DEFINE 1
#include "nccl_ofi_param.h"
