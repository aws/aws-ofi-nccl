/*
 * Copyright (c) 2024      Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi_log.h"
#include "nccl_ofi_param_impl.h"


#define OFI_NCCL_PARAM(type, name, env, default_value)						    \
	class ofi_nccl_param_impl<type> ofi_nccl_##name("OFI_NCCL_"  env, default_value);

std::forward_list<ofi_nccl_param_base *> ofi_nccl_parameter_list;

// This is ugly, but we define declaration versions of the macros before
// including nccl_ofi_param so that we have implemtations of the classes
// *somewhere*.  Note that it is very important that the parameter list object be
// created  before including this file, so that each object can register properly.
#include "nccl_ofi_param.h"

extern int ofi_nccl_parameters_init()
{
	for (auto iter = ofi_nccl_parameter_list.begin() ;
	     iter != ofi_nccl_parameter_list.end() ;
	     ++iter) {
		int ret = (*iter)->initialize();
		if (ret != 0) {
			return ret;
		}
	}

	return 0;
}
