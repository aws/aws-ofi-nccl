/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>

#include "test-common.hpp"
#include "nccl_ofi_ep_addr_list.h"

static void insert_addresses(nccl_ofi_ep_addr_list_t *ep_addr_list, size_t num_addr, int ep_num)
{
	for (size_t i = 0; i < num_addr; ++i) {
		nccl_net_ofi_ep_t *ep = NULL;
		int ret = nccl_ofi_ep_addr_list_get(ep_addr_list, &i, sizeof(i),
			&ep);
		if (ret != 0) {
			NCCL_OFI_WARN("nccl_ofi_ep_addr_list_get failed");
			exit(1);
		}

		if (i == 0) {
			if (ep) {
				NCCL_OFI_WARN("Ep unexpectedly returned");
				exit(1);
			}
			ret = nccl_ofi_ep_addr_list_insert(ep_addr_list,
				(nccl_net_ofi_ep_t *)(uintptr_t)(ep_num), &i, sizeof(i));
			if (ret != 0) {
				NCCL_OFI_WARN("nccl_ofi_ep_addr_list_insert failed");
				exit(1);
			}
		} else {
			if (!ep) {
				NCCL_OFI_WARN("No ep returned when expected. addr %ld, ep_num %d", i, ep_num);
				exit(1);
			}
			if ((uintptr_t)ep != (uintptr_t)ep_num) {
				NCCL_OFI_WARN("Unexpected ep returned");
			}
		}
	}
}

static int get_ep_for_addr(nccl_ofi_ep_addr_list_t *ep_addr_list, int addr_val)
{
	nccl_net_ofi_ep_t *ep = NULL;
	int ret = nccl_ofi_ep_addr_list_get(ep_addr_list, &addr_val,
					    sizeof(addr_val), &ep);
	if (ret) {
		NCCL_OFI_WARN("nccl_ofi_ep_addr_list_get failed");
		exit(1);
	}
	return (int)(uintptr_t)ep;
}

int main(int argc, char *argv[])
{
	ofi_log_function = logger;

	const size_t num_addr = 10;

	nccl_ofi_ep_addr_list_t *ep_addr_list =
		nccl_ofi_ep_addr_list_init(128);
	if (!ep_addr_list) {
		NCCL_OFI_WARN("Init ep addr list failed");
		exit(1);
	}

	/** Test insertion and retrieval **/
	insert_addresses(ep_addr_list, num_addr, 1);
	/** And again! **/
	insert_addresses(ep_addr_list, num_addr, 2);
	/** And again! **/
	insert_addresses(ep_addr_list, num_addr, 3);

	/* At this point, we have three eps (1-3), each of which have addresses (0-9) */

	/** Test delete nonexistant **/
	int r = 0;
	r = nccl_ofi_ep_addr_list_delete(ep_addr_list, (nccl_net_ofi_ep_t *)4); // (Doesn't exist)
	if (r != -ENOENT) {
		NCCL_OFI_WARN("Delete non-existing ep succeeded unexpectedly");
		exit(1);
	}

	/** Test delete middle **/
	r = nccl_ofi_ep_addr_list_delete(ep_addr_list, (nccl_net_ofi_ep_t *)2);
	if (r) {
		NCCL_OFI_WARN("Delete ep failed unexpectedly");
		exit(1);
	}

	/** Make sure eps 1 and 3 are still present */
	{
		int addr = num_addr;
		int ep = get_ep_for_addr(ep_addr_list, addr);
		if (ep != 1 && ep != 3) {
			NCCL_OFI_WARN("Unexpected ep from get_ep_for_addr");
			exit(1);
		}
		ep = get_ep_for_addr(ep_addr_list, addr);
		if (ep != 1 && ep != 3) {
			NCCL_OFI_WARN("Unexpected ep from get_ep_for_addr");
			exit(1);
		}
		/* Now both eps have this addr -- the next call should return NULL */
		ep = get_ep_for_addr(ep_addr_list, addr);
		if (ep != 0) {
			NCCL_OFI_WARN("Unexpected non-NULL ep");
			exit(1);
		}
	}

	/** Test delete front **/
	r = nccl_ofi_ep_addr_list_delete(ep_addr_list, (nccl_net_ofi_ep_t *)1);
	if (r) {
		NCCL_OFI_WARN("Delete ep failed unexpectedly");
		exit(1);
	}

	/** Make sure ep 3 is still present */
	{
		int addr = num_addr+1;
		int ep = get_ep_for_addr(ep_addr_list, addr);
		if (ep != 3) {
			NCCL_OFI_WARN("Unexpected ep from get_ep_for_addr");
			exit(1);
		}
		/* Now ep 3 has this addr -- the next call should return NULL */
		ep = get_ep_for_addr(ep_addr_list, addr);
		if (ep != 0) {
			NCCL_OFI_WARN("Unexpected non-NULL ep");
			exit(1);
		}
	}

	/** Test final delete **/
	r = nccl_ofi_ep_addr_list_delete(ep_addr_list, (nccl_net_ofi_ep_t *)3);
	if (r) {
		NCCL_OFI_WARN("Delete ep failed unexpectedly");
		exit(1);
	}

	/** Make sure get_ep_for_addr fails since there are no more endpoints **/
	{
		int addr = num_addr+2;
		int ep = get_ep_for_addr(ep_addr_list, addr);
		if (ep != 0) {
			NCCL_OFI_WARN("Unexpected non-NULL ep");
			exit(1);
		}
	}

	nccl_ofi_ep_addr_list_fini(ep_addr_list);

	printf("Test completed successfully!\n");

	return 0;
}
