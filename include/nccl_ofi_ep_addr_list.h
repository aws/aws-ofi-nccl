/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_EP_ADDR_LIST_H
#define NCCL_OFI_EP_ADDR_LIST_H

#include <cstdint>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>


/* Endpoint structure used by plugin code */
class nccl_net_ofi_ep_t;


class nccl_ofi_ep_addr_list_t {
public:
	/**
	 * Find endpoint in the list ep_pair_list that is not already connected to addr.
	 * If an endpoint is found, add this address to its connection list.
	 * If all endpoints are already connected to addr, return NULL.
	 *
	 * @param addr_in Libfabric address
	 * @param addr_size Size of address
	 * @param ep Output ep
	 *	     NULL if no ep found
	 *
	 * @return 0, on success
	 */
	int get(const void *addr_in, size_t addr_size, nccl_net_ofi_ep_t **ep);


        /**
	 * Add ep to the list ep_pair_list, with a single connection to addr.
	 *
	 * This function makes a copy of the data in addr, so the caller is free to
	 * modify the memory at addr as desired.
	 *
	 * @param ep pointer to endpoint
	 * @param addr_in Libfabric address of size MAX_EP_ADDR
	 *
	 * @return 0, on success
	 *	   -EINVAL, invalid argument
	 */
	int insert(nccl_net_ofi_ep_t *ep, const void *addr_in, size_t addr_size);


	/**
	 * Remove ep from the list ep_pair_list, if present
	 *
	 * @param ep pointer to endpoint
	 *
	 * @return 0, on success
	 *         -ENOENT, ep not found
	 */
	int remove(nccl_net_ofi_ep_t *ep);

private:
	// we need an object to represent opaque Libfabric addresses with two
	// different properties.  When we're searching for an address in the
	// address_set for each endpoint, we want a lightweight overlay of the
	// address passed into get() or insert().  But the definition of the
	// existing ep_addr_list interface is that the remote address memory may
	// be modified after insert() or get() returns.  This means that for the
	// key stored in address_set, we need to actually have a deep copy.  To
	// handle both cases, we create a surcture that can be initialized with
	// a memory region as a lightweight string_view overlay, and on copy
	// construction (which should only happen on insert into the
	// address_set), we make a deep copy to keep the API guarantee.
	//
	// Because of this behavior, data may be an empty vector.  view will
	// always point to useful data.
	class address_storage {
	public:
		address_storage(const void *addr, size_t addr_len);
		address_storage(const address_storage &other);

		bool operator==(const address_storage& rhs) const;
		const std::string_view& get_view(void) const { return view; }

	private:
		// ordering is important here.  view overlays at data, so data
		// must be initialized before creating the view in the copy
		// constructor.
                const std::vector<char> data;
		const std::string_view view;
	};

	struct address_storage_hash {
		std::size_t operator()(const address_storage& k) const {
			std::size_t val = std::hash<std::string_view>()(k.get_view());
			return val;
		}
	};

        using address_set = std::unordered_set<address_storage, address_storage_hash>;
	using endpoint_map = std::unordered_map<nccl_net_ofi_ep_t *, address_set>;

	std::mutex lock;
	endpoint_map endpoints;
};

#endif
