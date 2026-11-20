/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#if HAVE_GDRCOPY
#include <errno.h>
#include <stdint.h>
#include <gdrapi.h>

static int map_failure;
static int unpins;

void gdrapi_mock_set_map_failure(int value);
int gdrapi_mock_unpins(void);

void gdrapi_mock_set_map_failure(int value) { map_failure = value; unpins = 0; }
int gdrapi_mock_unpins(void) { return unpins; }

gdr_t gdr_open(void) { return (gdr_t)(uintptr_t)1; }
int gdr_close(gdr_t g) { return 0; }
int gdr_pin_buffer(gdr_t g, unsigned long addr, size_t size, uint64_t token,
		   uint32_t va_space, gdr_mh_t *handle)
{
	if (!map_failure) return EIO;
	handle->h = 42;
	return 0;
}
int gdr_unpin_buffer(gdr_t g, gdr_mh_t handle) { ++unpins; return 0; }
int gdr_map(gdr_t g, gdr_mh_t handle, void **ptr, size_t size) { return EIO; }
int gdr_unmap(gdr_t g, gdr_mh_t handle, void *ptr, size_t size) { return 0; }
int gdr_copy_to_mapping(gdr_mh_t handle, void *dst, const void *src, size_t n) { return EIO; }
int gdr_copy_from_mapping(gdr_mh_t handle, void *dst, const void *src, size_t n) { return EIO; }
void gdr_runtime_get_version(int *major, int *minor) { *major = 2; *minor = 4; }
int gdr_driver_get_version(gdr_t g, int *major, int *minor)
{
	*major = 2;
	*minor = 4;
	return 0;
}
#else
/* The test is skipped when the GDRCopy headers are not available. */
void gdrapi_mock_unavailable(void);
void gdrapi_mock_unavailable(void) {}
#endif
