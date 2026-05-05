/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi_ep_addr_list.h"
#include "nccl_ofi_log.h"


nccl_ofi_ep_addr_list_t::address_storage::address_storage(const void *addr,
                                                          size_t addr_len) :
	view(static_cast<const char*>(addr), addr_len)
{
}


nccl_ofi_ep_addr_list_t::address_storage::address_storage(const address_storage &other) :
	data(other.view.begin(), other.view.end()), view(data.data(), data.size())
{
}


bool nccl_ofi_ep_addr_list_t::address_storage::operator==(const address_storage &rhs) const
{
	return (view == rhs.view);
}


int nccl_ofi_ep_addr_list_t::get(const void *addr_in, size_t addr_size, nccl_net_ofi_ep_t **ep)
{
	address_storage remote_address = address_storage(addr_in, addr_size);
	std::lock_guard l(lock);

        *ep = NULL;

	for (auto ep_iter = endpoints.begin() ; ep_iter != endpoints.end() ; ++ep_iter) {
		if (0 == ep_iter->second.count(remote_address)) {
			/* found one that works */
			*ep = ep_iter->first;

			ep_iter->second.emplace(remote_address);

                        return 0;
		}
	}

        return 0;
}


int nccl_ofi_ep_addr_list_t::insert(nccl_net_ofi_ep_t *ep, const void *addr_in,
				    size_t addr_size)
{
	address_storage remote_address = address_storage(addr_in, addr_size);
	std::lock_guard l(lock);


	/* if the ep is already in the map, that's a problem. */
	if (endpoints.count(ep) != 0) {
		return -EINVAL;
	}

	auto ep_ret = endpoints.emplace(ep, address_set());
	if (!ep_ret.second) {
		NCCL_OFI_WARN("Failed to insert new endpoint");
		return -EINVAL;
	}

	auto endpoints_iter = ep_ret.first;
	auto addr_ret = endpoints_iter->second.emplace(remote_address);
	if (!addr_ret.second) {
		NCCL_OFI_WARN("Failed to insert address in new endpoint");
		endpoints.erase(ep_ret.first);
		return -EINVAL;
	}

	return 0;
}


int nccl_ofi_ep_addr_list_t::remove(nccl_net_ofi_ep_t *ep)
{
	std::lock_guard l(lock);

	return (endpoints.erase(ep) == 0) ? -ENOENT : 0;
}
