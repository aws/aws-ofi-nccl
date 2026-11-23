/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#if HAVE_GDRCOPY
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <new>

#include "unit_test.h"
#include "nccl_ofi_gdrcopy.h"

static std::atomic<long> live_allocations{0};

void *operator new(std::size_t size)
{
	void *ptr = std::malloc(size == 0 ? 1 : size);
	if (ptr == nullptr) throw std::bad_alloc();
	++live_allocations;
	return ptr;
}

void operator delete(void *ptr) noexcept
{
	if (ptr != nullptr) {
		--live_allocations;
		std::free(ptr);
	}
}

void operator delete(void *ptr, std::size_t) noexcept { ::operator delete(ptr); }

int main(int argc, char *argv[])
{
	if (argc != 2) return 1;
	const bool map_failure = std::strcmp(argv[1], "map") == 0;
	if (!map_failure && std::strcmp(argv[1], "pin") != 0) return 1;
	unit_test_init();

	void *lib = dlopen("libgdrapi.so", RTLD_NOW | RTLD_LOCAL);
	if (lib == nullptr) return 1;
	auto set_failure = reinterpret_cast<void (*)(int)>(dlsym(lib, "gdrapi_mock_set_map_failure"));
	auto unpins = reinterpret_cast<int (*)()>(dlsym(lib, "gdrapi_mock_unpins"));
	if (set_failure == nullptr || unpins == nullptr) return 1;
	set_failure(map_failure);

	{
		nccl_ofi_gdrcopy_ctx gdr;
		nccl_ofi_device_copy::RegHandle *handle = nullptr;
		const long before = live_allocations.load();
		const int ret = gdr.register_region(reinterpret_cast<void *>(0x10000), 4096, false, handle);
		const long after = live_allocations.load();
		if (ret != -EIO || handle != nullptr || unpins() != (map_failure ? 1 : 0) || after != before) {
			std::fprintf(stderr, "%s failure: ret=%d, handle=%p, unpins=%d, live before=%ld after=%ld\n",
				     argv[1], ret, static_cast<void *>(handle), unpins(), before, after);
			return 1;
		}
	}
	dlclose(lib);
	return 0;
}
#else
int main() { return 77; }
#endif
