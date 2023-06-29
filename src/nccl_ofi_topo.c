/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdbool.h>
#include <string.h>
#include <hwloc.h>
#include <rdma/fabric.h>

#include "nccl_ofi_log.h"
#include "nccl_ofi.h"
#include "nccl-headers/error.h"
#include "nccl_ofi_topo.h"

static const uint8_t display_controller_class_id = 0x03;
static const unsigned short nvidia_vendor_id = 0x10de;

/*
 * @brief Create vector of nccl_ofi_topo_data_t structs
 *
 * Allocate vector and initalize elements.
 *
 * @param	size
 *		Size of the vector
 * @return	allocated vector, if allocation succeeds
 *		NULL, on others
 */
static nccl_ofi_topo_data_vec_t *nccl_ofi_topo_data_vec_create(size_t size)
{
	nccl_ofi_topo_data_vec_t *vec = NULL;
	vec = (nccl_ofi_topo_data_vec_t*)calloc(1, sizeof(nccl_ofi_topo_data_vec_t));
	if (!vec) {
		NCCL_OFI_WARN("Failed to allocate struct nccl_ofi_topo_data_vec_t");
		return NULL;
	}

	if (size > 0) {
		vec->size = size;
		vec->data = (nccl_ofi_topo_data_t*)calloc(size, sizeof(nccl_ofi_topo_data_t));

		if (!vec->data) {
			free(vec);
			vec = NULL;
			NCCL_OFI_WARN("Failed to allocate array of struct nccl_ofi_topo_data_vec_t");
		}
	}

	return vec;
}

ncclResult_t nccl_ofi_topo_set_to_begin(nccl_ofi_topo_t *topo, nccl_ofi_topo_data_iterator_t *iter)
{
	if (!topo) {
		NCCL_OFI_WARN("Invalid NCCL OFI topology");
		return ncclInternalError;
	}

	if (!topo->data_vec) {
		NCCL_OFI_WARN("Invalid NCCL OFI topology user data");
		return ncclInternalError;
	}

	iter->begin = topo->data_vec->data;
	iter->end = topo->data_vec->data + topo->data_vec->size;

	return ncclSuccess;
}

/*
 * @brief	Return user data struct
 *
 * @return	user data, if available
 *		NULL, if iterator has reached end of vector
 */
static nccl_ofi_topo_data_t *nccl_ofi_get_user_data(nccl_ofi_topo_data_iterator_t *iter)
{
	if (!iter || iter->begin >= iter->end) return NULL;

	return iter->begin;
}

/*
 * @brief	Increment user data iterator
 */
static void nccl_ofi_inc_user_data_iter(nccl_ofi_topo_data_iterator_t *iter)
{
	if (!iter) return;
	++(iter->begin);
}

/*
 * @brief	Return pointer to fi_pci_attr struct of libfabric NIC info
 *
 * @return	Pointer to fi_pci_attr struct, if available
 *		NULL, on others
 */
static struct fi_pci_attr *ofi_info_get_pci_attr(struct fi_info *info) {

	struct fi_pci_attr *pci_addr = NULL;
	struct fi_bus_attr *bus = NULL;
	struct fid_nic *nic = info->nic;

	if (!nic) {
		return NULL;
	}

	bus = nic->bus_attr;
	if (!bus || bus->bus_type != FI_BUS_PCI) {
		return NULL;
	}

	pci_addr = &(bus->attr.pci);
	if (!pci_addr) {
		return NULL;
	}
	return pci_addr;
}

/*
 * @brief 	Test whether topology node represents an NVIDIA GPU PCI device
 */
static ncclResult_t is_nvidia_pci_dev(hwloc_obj_t obj, bool *res)
{
	uint8_t class_code;
	bool class_match;
	bool vendor_match;

	if (obj->type != HWLOC_OBJ_PCI_DEVICE) {
                *res = false;
                return ncclSuccess;
        }

        if (!obj->attr) {
                NCCL_OFI_WARN("Invalid hwloc object attribute pointer. Expected pointer to pcidev, but got NULL");
                return ncclInternalError;
        }

	/* The HWLOC class id is a 16 bit integer of format class
	   code:subclass, where each field is 8 bits. We only want
	   the class code. */
	class_code = obj->attr->pcidev.class_id >> 8;

	class_match = display_controller_class_id == class_code;
	vendor_match = obj->attr->pcidev.vendor_id == nvidia_vendor_id;

        *res = class_match && vendor_match;
        return ncclSuccess;
}

void nccl_ofi_topo_free(nccl_ofi_topo_t *topo)
{
	if (!topo) return;

	if (topo->topo) hwloc_topology_destroy(topo->topo);

	if (topo->data_vec) {
		nccl_ofi_topo_data_iterator_t data_iter;
		nccl_ofi_topo_set_to_begin(topo, &data_iter);

		/* Free libfabric NIC info lists */
		nccl_ofi_topo_data_t *data = nccl_ofi_get_user_data(&data_iter);
		nccl_ofi_inc_user_data_iter(&data_iter);
		while (data) {
			nccl_net_ofi_free_info_list(data->info_list);
			data = nccl_ofi_get_user_data(&data_iter);
			nccl_ofi_inc_user_data_iter(&data_iter);
		}

		/* Free data array and vector */
		free(topo->data_vec->data);
		free(topo->data_vec);
	}

	free(topo);
}

/*
 * brief	Enable I/O discovery for hardware topology
 *
 * Hwloc I/O discovery is disabled by default. This function enables
 * topology discovery of I/O devices.
 *
 * If I/O discovery is disabled, the hwloc creates a hardware topology
 * tree that does not include I/O devices as nodes. I/O devices are
 * hwloc OS devices, hwloc PCI devices, and hwloc bridges.
 *
 * Hwloc hostbridges are below normal hwloc objects (e.g., machine or
 * NUMA node). Children of hostbridges may be other hwloc bridges and
 * hwloc PCI devices. Hwloc OS devices may be children of PCI devices,
 * representing software handles such as 'eth0', 'sda', or 'cuda0'.
 *
 * If I/O discovery is enabled, Libfabric NICs are represented by
 * corresponding hwloc PCI devices. Hwloc may also add PCI devices to
 * represent GPUs.
 *
 * The libfabric NIC grouping algorithm uses the locality of NICs and
 * GPUs in the hardware topology to group NICs.
 *
 * This function is to be called after hardware topology is
 * initialized but before hardware topology is loaded.
 */
static void enable_hwloc_io_types(hwloc_topology_t topo)
{
	/* HWLOC API changes introduced in version 2.0.0 */
#if (HWLOC_API_VERSION >= 0x00020000)
	/* HWLOC_TOPOLOGY_FLAG_IO_DEVICES has been removed in favor of
	 * hwloc_topology_set_io_types_filter() with
	 * HWLOC_TYPE_FILTER_KEEP_ALL or HWLOC_TYPE_FILTER_KEEP_IMPORTANT */
	enum hwloc_type_filter_e filter = HWLOC_TYPE_FILTER_KEEP_ALL;
	hwloc_topology_set_io_types_filter(topo, filter);
#else
	unsigned long flags = hwloc_topology_get_flags(topo);
	flags |= HWLOC_TOPOLOGY_FLAG_IO_DEVICES;
	hwloc_topology_set_flags(topo, flags);
#endif
}

/*
 * brief	Return PCI device topology node corresponding to the
 *		libfabric NIC info struct
 *
 * A PCI topology node corresponds to the libfabric NIC info struct if
 * the topology node matches the bus ID reported by the libfabric NIC
 * info struct and if the topology node is of PCI class 02 (display
 * controller).
 *
 * @param	topo
 *		The topology
 * @param	info
 *		Libfabric NIC info struct
 * @return	topology node, if a corresponding topology node is found.
 *		NULL, on others
 * @return	ncclSuccess, if a corresponding topology node is found
 * 		or if no topology node is found for the bus ID reported by `info`
 *		ncclInternalError, on others
 */
static ncclResult_t get_hwloc_pcidev_by_fi_info(hwloc_topology_t topo,
						struct fi_info *info,
						hwloc_obj_t *obj) {
	*obj = NULL;
	hwloc_obj_t ret_obj = NULL;

	struct fi_pci_attr *attr = ofi_info_get_pci_attr(info);
	if (!attr) {
		NCCL_OFI_WARN("Failed to retrieve PCI attributes from NIC");
		return ncclInternalError;
	}

	ret_obj = hwloc_get_pcidev_by_busid(topo,
					    attr->domain_id,
					    attr->bus_id,
					    attr->device_id,
					    attr->function_id);

	if (!ret_obj) {
		/* Not finding a topology node corresponding to the
		 * info struct does not mean that something is going
		 * wrong. E.g., the NIC might just be removed manually from
		 * hwloc topology file. */
		return ncclSuccess;
	}

	if (!ret_obj->attr) {
		NCCL_OFI_WARN("Invalid hwloc object attribute pointer. Expected pointer to pcidev, but got NULL");
		return ncclInternalError;
	}

	*obj = ret_obj;
	return ncclSuccess;
}

/*
 * @brief	Return libfabric NIC info struct from info list that corresponds to input topology node
 *
 * @param	node
 *		The topology node
 * @param	info_list
 *		List of Libfabric NIC info structs
 * @return	Info struct that corresponds to topology node if found in info list
 *		NULL, otherwise
 * @return	ncclSuccess, on success
 *		non-zero, on error
 */
static ncclResult_t get_info_for_node(hwloc_obj_t node, struct fi_info *info_list, struct fi_info **ret_info)
{
	*ret_info = NULL;
	union hwloc_obj_attr_u *node_attr;
	struct fi_info *next;

	if (!info_list) {
		NCCL_OFI_WARN("No info list provided");
		return ncclInternalError;
	}

        node_attr = node->attr;
	if (!node_attr) {
		NCCL_OFI_WARN("Failed to retrieve attributes from hwloc topology node");
		return ncclInternalError;
	}

	if (node->type != HWLOC_OBJ_PCI_DEVICE) {
		return ncclSuccess;
	}

	/* Iterate through list, return info struct if found */
	next = info_list;
	do {
		struct fi_info *info = next;
		next = info->next;

		struct fi_pci_attr *attr = ofi_info_get_pci_attr(info);
		if (!attr) {
			NCCL_OFI_WARN("Failed to retrieve PCI attributes from NIC");
			return ncclInternalError;
		}

		if (node_attr->pcidev.domain == attr->domain_id &&
		    node_attr->pcidev.bus == attr->bus_id &&
		    node_attr->pcidev.dev == attr->device_id &&
		    node_attr->pcidev.func == attr->function_id) {
			*ret_info = info;
			return ncclSuccess;
		}

		/* Stop loop if end of list is reached or if info
		 * struct closes loop to list head */
	} while (next && next != info_list);

	return ncclSuccess;
}

/*
 * @brief	Count number of topology nodes that have a NIC or Nvidia GPU in its subtree
 *
 *
 * @param	topo
 *		Hwloc topology. Userdata pointers of topology nodes are expected to be set to NULL
 * @param	info_list
 *		Libfabric NIC info list to identify the NICs
 * @return	Number of nodes
 * @return	ncclSuccess, on success
 *		non-zero, on error
 */
static ncclResult_t count_nodes_with_gpu_or_nic_in_subtree(hwloc_topology_t topo,
							struct fi_info *info_list,
							int *count)
{
	ncclResult_t ret = ncclSuccess;
	hwloc_obj_t obj = NULL;

	while ((obj = hwloc_get_next_pcidev(topo, obj))) {
		bool is_gpu = false;
		struct fi_info *info;

		ret = is_nvidia_pci_dev(obj, &is_gpu);
		if (ret != ncclSuccess) {
			NCCL_OFI_WARN("Error while checking whether hwloc topology node is nvidia GPU");
			return ncclInternalError;
		}

		ret = get_info_for_node(obj, info_list, &info);
		if (ret != ncclSuccess) {
			NCCL_OFI_WARN("Error while retrieving libfabric NIC info struct corresponding to hwloc topology node");
			return ncclInternalError;
		}

		if (is_gpu || info) {
			/* Walk towards root, set counter and increment counter each time counter is set */
			hwloc_obj_t node = obj;
			while (node) {
				/* Skip node if this function is counting and
				 * if node it has already contributed to the
				 * counter (indicated by set user data
				 * pointer) */
				if (count && node->userdata) break;
				node->userdata = count;
				if (count) ++(*count);
				node = node->parent;
			}
		}

	}

	/* While counting, the function sets the user data pointer of
	 * the topology nodes to avoid counting nodes
	 * twice. Afterwards, invoke this function another time to
	 * clear the user data pointers. */
	if (count != NULL) return count_nodes_with_gpu_or_nic_in_subtree(topo, info_list, NULL);
	else return ncclSuccess;
}

/*
 * @brief	Walk to root from `node` and set user data
 *
 * Starting from input node `node`, walk upwards to the root and set
 * userdata. End ascend if a node is reached that already stores user
 * data. This likely means that another traversal from another node to
 * the root has already set the user data of the remaining nodes
 * towards the root. The user data is provided by `data_iter`.
 *
 * @param	node
 * 		Input topology node
 * @param	data_iter
 *		Data iterator from which the user data objects are extracted.
 * @return	ncclSuccess, on error
 *		ncclInternalError, if data iterator does not provide enough user data objects
 */
static ncclResult_t set_userdata_to_root(hwloc_obj_t node,
					 nccl_ofi_topo_data_iterator_t *data_iter)
{
	nccl_ofi_topo_data_t * user_data;

	/* Walk upwards to the root */
	while (node) {
		if (!node->userdata) {
			/* Abort when a node is reached that already stores user data */
			if (node->userdata) break;

			user_data = nccl_ofi_get_user_data(data_iter);
			if (!user_data) {
				NCCL_OFI_WARN("Failed to access user data of data_iter");
				return ncclInternalError;
			}
			node->userdata = user_data;
			user_data->node = node;
			nccl_ofi_inc_user_data_iter(data_iter);
		}
		node = node->parent;
	}

	return ncclSuccess;
}

/*
 * @brief	Add user data to topology nodes with a NIC or Nvidia GPU in its subtree.
 *		Also, add libfabric NIC info struct to NIC topology nodes.
 *
 * @param	ofi_topo
 *		NCCL OFI topology
 * @param	info_list
 *		List of libfabric NIC info structs used to identify topology nodes corresponding to NICs
 * @return
 */
static ncclResult_t set_user_data(nccl_ofi_topo_t *ofi_topo,
				  struct fi_info *info_list)
{
	ncclResult_t ret = ncclSuccess;
	hwloc_obj_t obj = NULL;
	nccl_ofi_topo_data_iterator_t data_iter;

	/* Retrieve number of topology nodes that have a Nvidia GPU or a NIC in their subtree */
	int num_nodes = 0;
	ret = count_nodes_with_gpu_or_nic_in_subtree(ofi_topo->topo, info_list, &num_nodes);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("Failed counting number of nodes that have a Nvidia GPU or NIC in their subtree.");
		return ncclInternalError;
	}

	/* Create vector that provides one user data struct for each
	 * topology node that has a Nvidia GPU or a NIC in its subtree */
	ofi_topo->data_vec = nccl_ofi_topo_data_vec_create(num_nodes);
	if (!ofi_topo->data_vec) {
		NCCL_OFI_WARN("Could not create user data vector.");
		return ncclInternalError;
	}
	nccl_ofi_topo_set_to_begin(ofi_topo, &data_iter);

	/* Iterate over all PCI topology nodes and find nodes
	 * corresponding to NICs and Nvidia GPUs. From those nodes,
	 * walk up towards the root and set user data. */
	while ((obj = hwloc_get_next_pcidev(ofi_topo->topo, obj))) {
		bool is_gpu = false;
		struct fi_info *info;

		ret = is_nvidia_pci_dev(obj, &is_gpu);
		if (ret != ncclSuccess) {
			NCCL_OFI_WARN("Error while checking whether hwloc topology node is nvidia GPU");
			return ncclInternalError;
		}

		ret = get_info_for_node(obj, info_list, &info);
		if (ret != ncclSuccess) {
			NCCL_OFI_WARN("Error while retrieving libfabric NIC info struct corresponding to hwloc topology node");
			return ncclInternalError;
		}

		if (is_gpu || info) {
			ret = set_userdata_to_root(obj, &data_iter);
			if (ret != ncclSuccess) {
				NCCL_OFI_WARN("Error while setting user data on path to root");
				return ncclInternalError;
			}
		}

		if (info) {
			/* Copy libfabric NIC info struct and store info struct in
			 * user data of topology node */
			nccl_ofi_topo_data_t *user_data = (nccl_ofi_topo_data_t *)obj->userdata;
			user_data->info_list = fi_dupinfo(info);
			if (!user_data->info_list) {
				NCCL_OFI_WARN("Unable to duplicate libfabric NIC info");
				return ncclInvalidArgument;
			}
		}

	}

	return ncclSuccess;
}

nccl_ofi_topo_t *nccl_ofi_topo_create(struct fi_info *info_list)
{
	ncclResult_t ret = ncclSuccess;

	/* Allocate NCCL OFI topology */
	nccl_ofi_topo_t *ofi_topo = calloc(1, sizeof(nccl_ofi_topo_t));
	if (!ofi_topo) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to allocate nccl_ofi_topo");
		ret = ncclInternalError;
		goto error;
	}

	/*
	 * Load hardware topology
	 */
	if (hwloc_topology_init(&ofi_topo->topo) != 0) {
		NCCL_OFI_WARN("Unable to initialize hardware topology.");
		goto error;
	}

	/* Prepare hardware topology ready to load IO nodes as well */
	enable_hwloc_io_types(ofi_topo->topo);
	if (hwloc_topology_load(ofi_topo->topo) != 0) {
		NCCL_OFI_WARN("Unable to load hardware topology.");
		goto error;
	}

	/* Add user data to topology nodes that have a nic or NVIDIA
	 * GPU in their subtree. Also, add libfabric NIC info structs
	 * to user data to topology nodes corresponding to the NICs. */
	ret = set_user_data(ofi_topo, info_list);
	if (ret != ncclSuccess) {
		NCCL_OFI_WARN("Data decoration failed.");
		goto error;
	}

	return ofi_topo;

 error:
	nccl_ofi_topo_free(ofi_topo);
	return NULL;
}

/*
 * @brief	Mark all topology nodes that store a libfabric NIC info
 *		struct in their subtrees
 */
static ncclResult_t mark_topo_nodes_with_ofi_info_subtree(nccl_ofi_topo_t *topo)
{
	nccl_ofi_topo_data_t *data = NULL;

	/* Iterate over user data that stores libfabric NIC info structs */
	nccl_ofi_topo_data_iterator_t data_iter;
	nccl_ofi_topo_set_to_begin(topo, &data_iter);
	while ((data = nccl_ofi_get_user_data(&data_iter))) {
		nccl_ofi_inc_user_data_iter(&data_iter);
		if (!data->info_list) {
			continue;
		}

		hwloc_obj_t obj = data->node;
		if (!obj) {
			NCCL_OFI_WARN("Expected initialized topology");
			return ncclInternalError;
		}

		/* Walk up topology tree up to root and mark topology nodes */
		while (obj) {
			nccl_ofi_topo_data_t *obj_data = (nccl_ofi_topo_data_t *)obj->userdata;
			if (!obj_data) {
				NCCL_OFI_WARN("Invalid user data pointer");
				return ncclInternalError;
			}

			obj_data->is_nic_subtree = true;
			obj = obj->parent;
		}
	}

	return ncclSuccess;
}

/*
 * @brief	Walk up the tree until we find a subtree that has a
 *		libfabric NIC info struct (marked topology node) in its subtree
 *		and increase its group count by one
 *
 * @param 	node
 * 		Topology node on which this operation is started
 */
static ncclResult_t propagate_gpu_count(hwloc_obj_t node)
{
	nccl_ofi_topo_data_t *userdata = (nccl_ofi_topo_data_t *)node->userdata;
	if (!userdata) {
		NCCL_OFI_WARN("Invalid user data pointer");
		return ncclInternalError;
	}

	if (userdata->contributed_gpu) return ncclSuccess;
	userdata->contributed_gpu = true;

	/* Walk towards root */
	while (node) {
		userdata = (nccl_ofi_topo_data_t *)node->userdata;
		if (!userdata) {
			NCCL_OFI_WARN("Invalid user data pointer");
			return ncclInternalError;
		}

		/* Node found. Increase group count. */
		if (userdata->is_nic_subtree) {
			userdata->num_groups++;
			break;
		}

		node = node->parent;
	}

	return ncclSuccess;
}

/*
 * @brief	Propagate GPU counts from NVIDIA topology nodes to marked topology nodes.
 */
static ncclResult_t propagate_gpu_group_counts(hwloc_topology_t topo)
{
	ncclResult_t ret = ncclSuccess;
	hwloc_obj_t obj = NULL;

	/* Iterate over all PCI topology nodes and find nodes
	 * corresponding to NICs and Nvidia GPUs. From those nodes,
	 * walk up towards the root and set user data. */
	while ((obj = hwloc_get_next_pcidev(topo, obj))) {
		bool is_gpu = false;

		ret = is_nvidia_pci_dev(obj, &is_gpu);
		if (ret != ncclSuccess) {
			NCCL_OFI_WARN("Error while checking whether hwloc topology node is nvidia GPU");
			return ncclInternalError;
		}

		if (is_gpu) {
			propagate_gpu_count(obj);
		}
	}

	return ncclSuccess;
}

/*
 * @brief	Lift libfabric NIC info objects, stored in the user data of
 *		topology nodes, up to nodes with group count of one or more
 */
static ncclResult_t lift_up_ofi_infos(nccl_ofi_topo_t *topo)
{
	nccl_ofi_topo_data_t *data = NULL;
	nccl_ofi_topo_data_t *user_data = NULL;

	/* Iterate over user data. Since user data is added to all
	 * topology nodes that have a "NIC topology nodes" or a
	 * "Nvidia GPU topology nodes" it their subtree, all info
	 * structs are found. */
	nccl_ofi_topo_data_iterator_t data_iter;
	nccl_ofi_topo_set_to_begin(topo, &data_iter);
	while ((data = nccl_ofi_get_user_data(&data_iter))) {
		nccl_ofi_inc_user_data_iter(&data_iter);
		if (!data->info_list) {
			continue;
		}

		hwloc_obj_t obj = data->node;
		if (!obj) {
			NCCL_OFI_WARN("Expected initialized topology");
			return ncclInternalError;
		}

		user_data = (nccl_ofi_topo_data_t *)obj->userdata;

		/* Info object is already stored at appropriate node */
		if (user_data->num_groups > 0) continue;

		/* Search for topology nodes towards the root with a
		 * group count of one or more and add libfabric NIC
		 * info list to that node. */
		while (obj) {
			nccl_ofi_topo_data_t *user_data = (nccl_ofi_topo_data_t *)obj->userdata;
			if (!user_data) {
				NCCL_OFI_WARN("Invalid user data pointer");
				return ncclInternalError;
			}

			if (user_data->num_groups > 0) {
				/* Find end of list */
				struct fi_info *list_end = data->info_list;
				while(list_end->next) {
					list_end = list_end->next;
				}

				/* Concatenate lists */
				list_end->next = user_data->info_list;
				user_data->info_list = data->info_list;
				data->info_list = NULL;
				break;
			}
			obj = obj->parent;
			if (!obj) {
				NCCL_OFI_WARN("Unable to attach NIC to GPU.");
				return ncclInternalError;
			}
		}
	}

	return ncclSuccess;
}

/*
 * @brief	Split NIC info list into 'num_groups' and add each group to the
 *		topology node corresponding to its leader (first NIC of the list).
 *
 * @param	topo
 *		The topology
 * @param	info_list
 *		Libfabric NIC info list
 * @param	num_group
 *		Number of groups to create
 *
 * @return	ncclSuccess, on success
 * 		ncclInternalError, on others
 */
static ncclResult_t create_groups_from_info_list(nccl_ofi_topo_t *topo,
						 struct fi_info **info_list,
						 int num_groups)
{
	ncclResult_t ret = ncclSuccess;
	int group_idx = 0;

	/* Calculate length of input list */
	int size = 0;
	{
		struct fi_info *curr = *info_list;
		while (curr) {
			curr = curr->next;
			++size;
		}
	}

	/* Adjust number of groups if input list does not provide enough members */
	num_groups = num_groups < size ? num_groups : size;
	/* Number of groups with one additional member. Handles the
	 * case where list size is not a multiple of number of
	 * groups */
	const int num_large_groups = size % num_groups;
	int group_size = size / num_groups + 1;

	for (; group_idx < num_groups; ++group_idx) {
		hwloc_obj_t obj;
		/* If the number of NIC infos is not a multiple of
		 * group size, latter candidates have one candidate
		 * less. */
		if (group_idx == num_large_groups) --group_size;
		if (group_size == 0) break;

		/* Retrieve topology node of leader */
		ret = get_hwloc_pcidev_by_fi_info(topo->topo, *info_list, &obj);
		if (ret != ncclSuccess) {
			NCCL_OFI_WARN("Retrieval of topology node corresponding to libfabric NIC failed with error");
			break;
		}
		if (!obj) {
			NCCL_OFI_WARN("hwloc failed detecting PCI NIC info.");
			ret = ncclInternalError;
			break;
		}

		nccl_ofi_topo_data_t *user_data = (nccl_ofi_topo_data_t *)obj->userdata;
		if (!user_data) {
			NCCL_OFI_WARN("Invalid user data pointer");
			return ncclInternalError;
		}

		if (user_data->info_list == *info_list) {
			if (group_idx + 1 == num_groups) {
				break;
			} else {
				NCCL_OFI_WARN("Invalid state of topology. "
					      "This state should not be reached.");
				return ncclInternalError;
			}
		}

		/* Add list topology node */
		user_data->info_list = *info_list;

		/* Cut list into two lists after group size list elements */
		struct fi_info *end = user_data->info_list;
		int i = 1;
		for (; i < group_size; ++i) {
			end = end->next;
		}

		/* Move list remainder to input list */
		*info_list = end->next;
		end->next = NULL;
	}

	return ret;
}

/*
 * @brief	Split libfabric NIC info lists of topology nodes with 'num_groups' > 0
 *		into 'num_groups' lists (groups) and add these lists
 *		to the corresponding topology nodes of their leaders (first
 *		NIC of the list).
 *
 * @return	ncclSuccess, on success
 * 		ncclInternalError, on others
 */
static ncclResult_t create_groups_from_info_lists(nccl_ofi_topo_t *topo)
{
	nccl_ofi_topo_data_t *data = NULL;

	/* Iterate over user data of topology nodes */
	nccl_ofi_topo_data_iterator_t data_iter;
	nccl_ofi_topo_set_to_begin(topo, &data_iter);
	while ((data = nccl_ofi_get_user_data(&data_iter))) {
		nccl_ofi_inc_user_data_iter(&data_iter);
		if (!data->info_list) {
			continue;
		}

		if (data->num_groups == 0) {
			continue;
		}

		struct fi_info *info_list = data->info_list;
		data->info_list = NULL;
		int num_groups = data->num_groups;
		data->num_groups = 0;

		/* Create groups from list */
		ncclResult_t ret = create_groups_from_info_list(topo, &info_list, num_groups);
		if (ret != ncclSuccess) {
			data->info_list = info_list;
			return ret;
		}
	}

	return ncclSuccess;
}

/*
 * @brief	Print libfabric NIC info lists stored in user data of topology nodes
*/
static void print_nic_groups(nccl_ofi_topo_t *topo) {
	nccl_ofi_topo_data_t *data = NULL;
	nccl_ofi_topo_data_iterator_t data_iter;
	nccl_ofi_topo_set_to_begin(topo, &data_iter);

	int group_idx = 0;
	while ((data = nccl_ofi_get_user_data(&data_iter))) {
		nccl_ofi_inc_user_data_iter(&data_iter);
		if (!data->info_list) {
			continue;
		}

		struct fi_info *info = data->info_list;
		int info_idx = 0;
		while (info) {
			struct fi_pci_attr *attr = ofi_info_get_pci_attr(info);
			if (attr) {
				NCCL_OFI_INFO(NCCL_INIT,
					      "NIC group %i device #%i "
					      "%04x:%02x:%02x.%01x",
					      group_idx, info_idx,
					      attr->domain_id,
					      attr->bus_id,
					      attr->device_id,
					      attr->function_id);
			}
			++info_idx;
			info = info->next;
		}
		++group_idx;
	}
}

ncclResult_t nccl_ofi_topo_group(nccl_ofi_topo_t *topo)
{
	ncclResult_t ret = ncclSuccess;

        ret = mark_topo_nodes_with_ofi_info_subtree(topo);
	if (ret != ncclSuccess) {
		return ret;
	}

	ret = propagate_gpu_group_counts(topo->topo);
	if (ret != ncclSuccess) {
		return ret;
	}
	ret = lift_up_ofi_infos(topo);
	if (ret != ncclSuccess) {
		return ret;
	}

	ret = create_groups_from_info_lists(topo);
	if (ret != ncclSuccess) {
		return ret;
	}

	print_nic_groups(topo);
	return ret;
}

ncclResult_t nccl_ofi_topo_num_info_lists(nccl_ofi_topo_t *topo, int *num_lists)
{
	if (!topo || !topo->data_vec) {
		NCCL_OFI_WARN("Invalid topology. Topology is not initialized.");
		return ncclInvalidArgument;
	}

	nccl_ofi_topo_data_t *data = NULL;
	nccl_ofi_topo_data_iterator_t data_iter;
	nccl_ofi_topo_set_to_begin(topo, &data_iter);

	*num_lists = 0;
	while ((data = nccl_ofi_get_user_data(&data_iter))) {
		nccl_ofi_inc_user_data_iter(&data_iter);
		*num_lists += (!data->info_list) ? 0 : 1;
	}

	return ncclSuccess;
}

struct fi_info *nccl_ofi_topo_next_info_list(nccl_ofi_topo_data_iterator_t *iter)
{
	if (!iter || iter->begin >= iter->end) return NULL;

	struct fi_info *info_list = NULL;
	nccl_ofi_topo_data_t *data = NULL;

	while ((data = nccl_ofi_get_user_data(iter))) {
		nccl_ofi_inc_user_data_iter(iter);
		if (data->info_list) {
			return data->info_list;
		}
	}

	return info_list;
}
