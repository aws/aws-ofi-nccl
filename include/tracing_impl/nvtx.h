/*
 * Copyright (c) 2022-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#pragma once
#if HAVE_NVTX_TRACING
#include "nvToolsExt.h"
static inline void nvtx_push(const char* name) {
	const nvtxEventAttributes_t eventAttrib = {
		.version = NVTX_VERSION,
		.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE,
		.colorType = NVTX_COLOR_ARGB,
		.color = 0xeb9234,
		.messageType = NVTX_MESSAGE_TYPE_ASCII,
		.message = { .ascii = name },
	};
	nvtxRangePushEx(&eventAttrib);
}
static inline void nvtx_pop(void) {
	nvtxRangePop();
}
#else
static inline void nvtx_push(const char* name){ (void)name; }
static inline void nvtx_pop(void){}
#endif
