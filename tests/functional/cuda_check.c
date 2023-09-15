/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

/*
 * This test is the only functional test that does not directly link
 * against libcudart.  Its purpose is to ensure that that the build
 * will fail if a direct cuda call is made from the plugin.
 */

#include "config.h"

#include "nccl_ofi_api.h"

void logger(ncclDebugLogLevel level, unsigned long flags, const char *filefunc,
	    int line, const char *fmt, ...)
{
}

int main(int argc, char* argv[])
{
	return nccl_net_ofi_init(logger);
}
