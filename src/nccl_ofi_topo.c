/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdbool.h>
#include <string.h>
#include <hwloc.h>
#include <rdma/fabric.h>
#include <errno.h>
#include <stdlib.h>
#include <assert.h>

#include "nccl_ofi_log.h"
#include "nccl_ofi.h"
#include "nccl-headers/error.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_math.h"

#if HAVE_CUDA
static const uint8_t target_class_id = 0x03;		/* Display controller class */
static const unsigned short target_vendor_id = 0x10de;	/* NVIDIA */
#else
static const uint8_t target_class_id = 0x08;		/* System peripheral */
static const unsigned short target_vendor_id = 0x1d0f;	/* Amazon */
#endif

/* Maximum length of the device property read from file by function
 * get_device_property() */
#define MAX_DEV_PROPERTY_LENGTH 16

const char *speed_name = "max_link_speed";
const char *width_name = "max_link_width";

/* `pcie_gen[i]` defines the speed of a PCIe lane of PCIe generation `i+1` */
const char *pcie_gen[] = {"2.5", "5", "8", "16", "32", "64"};

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

int nccl_ofi_topo_set_to_begin(nccl_ofi_topo_t *topo, nccl_ofi_topo_data_iterator_t *iter)
{
	if (!topo) {
		NCCL_OFI_WARN("Invalid NCCL OFI topology");
		return -EINVAL;
	}

	if (!topo->data_vec) {
		NCCL_OFI_WARN("Invalid NCCL OFI topology user data");
		return -EINVAL;
	}

	iter->begin = topo->data_vec->data;
	iter->end = topo->data_vec->data + topo->data_vec->size;

	return 0;
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
static int is_accelerator_dev(hwloc_obj_t obj, bool *res)
{
	uint8_t class_code;
	bool class_match;
	bool vendor_match;

	if (obj->type != HWLOC_OBJ_PCI_DEVICE) {
                *res = false;
                return 0;
        }

        if (!obj->attr) {
                NCCL_OFI_WARN("Invalid hwloc object attribute pointer. Expected pointer to pcidev, but got NULL");
                return -EINVAL;
        }

	/* The HWLOC class id is a 16 bit integer of format class
	   code:subclass, where each field is 8 bits. We only want
	   the class code. */
	class_code = obj->attr->pcidev.class_id >> 8;

	/*
	 * TODO: This is still a broad match that assumes any Amazon device
	 * registered with class "System Peripheral" is a Neuron device.  While
	 * this is true today, it might not be in the future.  Filtering on this
	 * is better than statically matching against the supported device IDs,
	 * which we would have to manually update as newer generations get released.
	 * In the future, we should update this to dynamically query Neuron
	 * devices on the instance and match the hwloc node against the
	 * discovered Neuron device BDFs.
	 */
	class_match = target_class_id == class_code;
	vendor_match = obj->attr->pcidev.vendor_id == target_vendor_id;

        *res = class_match && vendor_match;
        return 0;
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
	/*
	 * HWLOC_TOPOLOGY_FLAG_WHOLE_IO has been removed in favor of
	 * hwloc_topology_set_io_types_filter() with HWLOC_TYPE_FILTER_KEEP_ALL
	 * or HWLOC_TYPE_FILTER_KEEP_IMPORTANT
	 */
	enum hwloc_type_filter_e filter = HWLOC_TYPE_FILTER_KEEP_ALL;
	hwloc_topology_set_io_types_filter(topo, filter);
#else
	unsigned long flags = hwloc_topology_get_flags(topo);
	/*
	 * We want to detect I/O devices, DMA buses, and bridges in the PCIe
	 * topology in addition to the default system components such as CPU,
	 * memory, etc.
	 */
	flags |= HWLOC_TOPOLOGY_FLAG_WHOLE_IO;
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
 * @return	0, if a corresponding topology node is found
 * 		or if no topology node is found for the bus ID reported by `info`
 *		ncclInternalError, on others
 */
static int get_hwloc_pcidev_by_fi_info(hwloc_topology_t topo,
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
		return 0;
	}

	if (!ret_obj->attr) {
		NCCL_OFI_WARN("Invalid hwloc object attribute pointer. Expected pointer to pcidev, but got NULL");
		return ncclInternalError;
	}

	*obj = ret_obj;
	return 0;
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
 * @return	0, on success
 *		non-zero, on error
 */
static int get_info_for_node(hwloc_obj_t node, struct fi_info *info_list, struct fi_info **ret_info)
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
		return 0;
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
			return 0;
		}

		/* Stop loop if end of list is reached or if info
		 * struct closes loop to list head */
	} while (next && next != info_list);

	return 0;
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
 * @return	0, on success
 *		non-zero, on error
 */
static int count_nodes_with_accel_or_nic_in_subtree(hwloc_topology_t topo,
							   struct fi_info *info_list,
							   int *count)
{
	int ret = 0;
	hwloc_obj_t obj = NULL;

	while ((obj = hwloc_get_next_pcidev(topo, obj))) {
		bool is_accel = false;
		struct fi_info *info;

		ret = is_accelerator_dev(obj, &is_accel);
		if (ret != 0) {
			NCCL_OFI_WARN("Error while checking whether hwloc topology node is nvidia GPU");
			return ncclInternalError;
		}

		ret = get_info_for_node(obj, info_list, &info);
		if (ret != 0) {
			NCCL_OFI_WARN("Error while retrieving libfabric NIC info struct corresponding to hwloc topology node");
			return ncclInternalError;
		}

		if (is_accel || info) {
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
	if (count != NULL) return count_nodes_with_accel_or_nic_in_subtree(topo, info_list, NULL);
	else return 0;
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
 * @return	0, on error
 *		ncclInternalError, if data iterator does not provide enough user data objects
 */
static int set_userdata_to_root(hwloc_obj_t node,
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

	return 0;
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
static int set_user_data(nccl_ofi_topo_t *ofi_topo,
				  struct fi_info *info_list)
{
	int ret = 0;
	hwloc_obj_t obj = NULL;
	nccl_ofi_topo_data_iterator_t data_iter;

	/* Retrieve number of topology nodes that have a Nvidia GPU or a NIC in their subtree */
	int num_nodes = 0;
	ret = count_nodes_with_accel_or_nic_in_subtree(ofi_topo->topo, info_list, &num_nodes);
	if (ret != 0) {
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
		bool is_accel = false;
		struct fi_info *info;

		ret = is_accelerator_dev(obj, &is_accel);
		if (ret != 0) {
			NCCL_OFI_WARN("Error while checking whether hwloc topology node is nvidia GPU");
			return ncclInternalError;
		}

		ret = get_info_for_node(obj, info_list, &info);
		if (ret != 0) {
			NCCL_OFI_WARN("Error while retrieving libfabric NIC info struct corresponding to hwloc topology node");
			return ncclInternalError;
		}

		if (is_accel || info) {
			ret = set_userdata_to_root(obj, &data_iter);
			if (ret != 0) {
				NCCL_OFI_WARN("Error while setting user data on path to root");
				return ncclInternalError;
			}
		}

		if (info) {
			/* Copy libfabric NIC info struct and store info struct in
			 * user data of topology node */
			nccl_ofi_topo_data_t *user_data = (nccl_ofi_topo_data_t *)obj->userdata;
			user_data->info_list = fi_dupinfo(info);
			user_data->info_list_len = 1;

			if (!user_data->info_list) {
				NCCL_OFI_WARN("Unable to duplicate libfabric NIC info");
				return ncclInvalidArgument;
			}

			ofi_topo->max_group_size = 1;
		}

	}

	return 0;
}

nccl_ofi_topo_t *nccl_ofi_topo_create(struct fi_info *info_list)
{
	int ret = 0;

	/* Allocate NCCL OFI topology */
	nccl_ofi_topo_t *ofi_topo = calloc(1, sizeof(nccl_ofi_topo_t));
	if (!ofi_topo) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to allocate nccl_ofi_topo");
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
	if (ret != 0) {
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
static int mark_topo_nodes_with_ofi_info_subtree(nccl_ofi_topo_t *topo)
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

	return 0;
}

/*
 * @brief	Walk up the tree until we find a subtree that has a
 *		libfabric NIC info struct (marked topology node) in its subtree
 *		and increase its group count by one
 *
 * @param 	gpu_node
 * 		Topology node on which this operation is started
 */
static int propoagate_accel_count(hwloc_obj_t gpu_node)
{
	hwloc_obj_t node = gpu_node;
	nccl_ofi_topo_data_t *userdata = (nccl_ofi_topo_data_t *)node->userdata;
	if (!userdata) {
		NCCL_OFI_WARN("Invalid user data pointer");
		return ncclInternalError;
	}

	if (userdata->contributed_gpu) return 0;
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
			userdata->gpu_group_node = gpu_node;
			break;
		}

		node = node->parent;
	}

	return 0;
}

/*
 * @brief	Propagate counts from accelerator topology nodes to marked topology nodes.
 */
static int propoagate_accel_group_counts(hwloc_topology_t topo)
{
	int ret = 0;
	hwloc_obj_t obj = NULL;

	/* Iterate over all PCI topology nodes and find nodes
	 * corresponding to NICs and Nvidia GPUs or Amazon Neuron devices. From
	 * those nodes, walk up towards the root and set user data. */
	while ((obj = hwloc_get_next_pcidev(topo, obj))) {
		bool is_accel = false;

		ret = is_accelerator_dev(obj, &is_accel);
		if (ret != 0) {
			NCCL_OFI_WARN("Error while checking whether hwloc topology node is an accelerator");
			return ncclInternalError;
		}

		if (is_accel) {
			propoagate_accel_count(obj);
		}
	}

	return 0;
}

/*
 * @brief	Lift libfabric NIC info objects, stored in the user data of
 *		topology nodes, up to nodes with group count of one or more
 */
static int lift_up_ofi_infos(nccl_ofi_topo_t *topo)
{
	nccl_ofi_topo_data_t *source_data = NULL;
	nccl_ofi_topo_data_t *target_data = NULL;

	/* Iterate over user data. Since user data is added to all
	 * topology nodes that have a "NIC topology nodes" or a
	 * "accelerator topology nodes" it their subtree, all info
	 * structs are found. */
	nccl_ofi_topo_data_iterator_t data_iter;
	nccl_ofi_topo_set_to_begin(topo, &data_iter);
	while ((source_data = nccl_ofi_get_user_data(&data_iter))) {
		nccl_ofi_inc_user_data_iter(&data_iter);
		if (!source_data->info_list) {
			continue;
		}

		hwloc_obj_t target_obj = source_data->node;
		if (!target_obj) {
			NCCL_OFI_WARN("Expected initialized topology");
			return ncclInternalError;
		}

		target_data = (nccl_ofi_topo_data_t *)target_obj->userdata;

		/* Info object is already stored at appropriate node */
		if (target_data->num_groups > 0) continue;

		/* Search for topology nodes towards the root with a
		 * group count of one or more and add libfabric NIC
		 * info list to that node. */
		while (target_obj) {
			target_data = (nccl_ofi_topo_data_t *)target_obj->userdata;
			if (!target_data) {
				NCCL_OFI_WARN("Invalid user data pointer");
				return ncclInternalError;
			}

			if (target_data->num_groups > 0) {
				/* Find end of list */
				struct fi_info *list_end = source_data->info_list;
				while(list_end->next) {
					list_end = list_end->next;
				}

				/* Concatenate lists */
				list_end->next = target_data->info_list;
				target_data->info_list = source_data->info_list;
				target_data->info_list_len += source_data->info_list_len;
				source_data->info_list = NULL;
				source_data->info_list_len = 0;
				break;
			}
			target_obj = target_obj->parent;
			if (!target_obj) {
				NCCL_OFI_WARN("Unable to attach NIC to accelerator.");
				return ncclInternalError;
			}
		}
	}

	return 0;
}

/*
 * @brief	Split NIC info list into 'num_groups' and add each group to the
 *		topology node corresponding to its leader (first NIC of the list).
 *
 * @param	topo
 *		The topology
 * @param	info_list
 *		Libfabric NIC info list
 * @param	num_infos
 *		Length of `info_list`
 * @param	gpu_group_node
 *		One GPU topology node that is the closest to the NICs in `info_list`
 * @param	num_group
 *		Number of groups to create
 *
 * @return	0, on success
 * 		ncclInternalError, on others
 */
static int create_groups_from_info_list(nccl_ofi_topo_t *topo,
						 struct fi_info **info_list,
						 int num_infos,
						 hwloc_obj_t gpu_group_node,
						 int num_groups)
{
	int ret = 0;
	int group_idx = 0;

	/* Adjust number of groups if input list does not provide enough members */
	num_groups = NCCL_OFI_MIN(num_groups, num_infos);
	/* Number of groups with one additional member. Handles the
	 * case where list size is not a multiple of number of
	 * groups */
	const int num_large_groups = num_infos % num_groups;
	int group_size = num_infos / num_groups + 1;

	for (; group_idx < num_groups; ++group_idx) {
		hwloc_obj_t obj;
		/* If the number of NIC infos is not a multiple of
		 * group size, latter candidates have one candidate
		 * less. */
		if (group_idx == num_large_groups) --group_size;
		if (group_size == 0) break;

		/* Retrieve topology node of leader */
		ret = get_hwloc_pcidev_by_fi_info(topo->topo, *info_list, &obj);
		if (ret != 0) {
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
		user_data->info_list_len = group_size;
		user_data->gpu_group_node = gpu_group_node;

		/* Track maximum group size */
		topo->max_group_size = NCCL_OFI_MAX(topo->max_group_size, group_size);

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
 * @return	0, on success
 * 		ncclInternalError, on others
 */
static int create_groups_from_info_lists(nccl_ofi_topo_t *topo)
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
		int info_list_len = data->info_list_len;
		data->info_list_len = 0;
		int num_groups = data->num_groups;
		data->num_groups = 0;

		/* Create groups from list */
		int ret = create_groups_from_info_list(topo,
								&info_list,
								info_list_len,
								data->gpu_group_node,
								num_groups);
		if (ret != 0) {
			data->info_list = info_list;
			return ret;
		}
	}

	return 0;
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

int nccl_ofi_topo_group(nccl_ofi_topo_t *topo)
{
	int ret = 0;

        ret = mark_topo_nodes_with_ofi_info_subtree(topo);
	if (ret != 0) {
		return ret;
	}

	ret = propoagate_accel_group_counts(topo->topo);
	if (ret != 0) {
		return ret;
	}
	ret = lift_up_ofi_infos(topo);
	if (ret != 0) {
		return ret;
	}

	ret = create_groups_from_info_lists(topo);
	if (ret != 0) {
		return ret;
	}

	print_nic_groups(topo);
	return ret;
}

/*
 * @brief	Return NUMA node child from memory children list of `node`
 *
 *  Since HWLOC 2.0, NUMA topology nodes are stored in a specific
 *  memory children list. Scan this list and return the first NUMA
 *  child node that is found that does not store user data.
 *
 * @param	node
 *		Topology node whose memory children list is to be scanned
 * @return	NUMA node topology node, if found
 *		NULL, otherwise
 */
static hwloc_obj_t get_numa_mem_child(hwloc_obj_t node)
{
	hwloc_obj_t child = NULL;

#if (HWLOC_API_VERSION >= 0x00020000)
	/* Recurse on memory children */
	if (node->memory_arity > 0) {
		child = node->memory_first_child;
		while (child != NULL) {
			nccl_ofi_topo_data_t *topo_data = (nccl_ofi_topo_data_t *)child->userdata;
			if (child->type == HWLOC_OBJ_NUMANODE && topo_data == NULL) {
				break;
			}
			child = child->next_sibling;
		}
	}
#endif

	return child;
}

/* 
 * @brief	Return PCI device property of PCI device
 *
 * This function reads first `MAX_DEV_PROPERTY_LENGTH` characters from
 * device property file
 * `/sys/bus/pci/devices/{domain}:{bus}:{dev}.{func}/{prop_name}`. Reading
 * may stop after a newline character is read or if the file
 * ends.
 *
 * @param	domain
 *		Domain of the PCI BUS ID
 * @param	bus
 *		Bus of the PCI BUS ID
 * @param	dev
 *		Device of the PCI BUS ID
 * @param	func
 *		Function of the PCI BUS ID
 * @param	prop_name
 *		File name of the device property
 * @return	Pointer to an element of a char array to write device property to.
 *		The array has to be allocated by the caller of this function.
 *		There must be space for at least `MAX_DEV_PROPERTY_LENGTH` 
 *		characters in addition to the delimiting `\0`.
 * @return	0, on sucess
 *		non-zero, on error
 */
static int get_device_property(unsigned domain, unsigned bus,
			       unsigned dev, unsigned func,
			       const char *prop_name, char *prop)
{
	int ret = 0;
	FILE *file;
	char *path_format = "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/%s";
	size_t path_len;
	char *path = NULL;

        if ((path_len = snprintf(NULL, 0, path_format, domain, bus, dev, func, prop_name)) < 0) {
		NCCL_OFI_WARN("Failed to determine device property path length of property %s. ERROR: %s",
			      prop_name, strerror(errno));
		ret = -errno;
		goto error;
	}
	path = malloc(path_len + 1);
	if (!path) {
		NCCL_OFI_WARN("Device property file path malloc failed: %s", strerror(errno));
		ret = -ENOMEM;
		goto error;
	}

	/* Create file path */
	if (snprintf(path, path_len + 1, path_format, domain, bus, dev, func, prop_name) < 0) {
		NCCL_OFI_WARN("Failed to create device property path for property %s. ERROR: %s",
			      prop_name, strerror(errno));
		ret = -errno;
		goto error;
	}

	/* Open file and read property */
	if ((file = fopen(path, "r")) != NULL) {
		char *rc = fgets(prop, MAX_DEV_PROPERTY_LENGTH + 1, file);
		if (feof(file) && !rc) {
			/* No bytes has been read. Let caller decide
			 * whether this is an error or not. */
			prop[0] = '\0';
		} else if (ferror(file)) {
			NCCL_OFI_WARN("Failed to read device property file %s. ERROR: %s",
				      path, strerror(errno));
			ret = -errno;
			goto exit;
		}
	} else {
		NCCL_OFI_WARN("Failed to open device property file %s. ERROR: %s",
			      path, strerror(errno));
		ret = -errno;
		goto error;
	}

 exit:
	if (fclose(file)) {
		NCCL_OFI_WARN("Failed to close device property file %s. ERROR: %s",
			      path, strerror(errno));
		ret = -errno;
	}
 error:
	if (path) free(path);

	return ret;
}

/*
 * @brief	Read link speed and width of PCI device or bridge from file system
 *
 * @param	node
 *		PCI device or bridge topology node
 * @param	is_nic
 *		True if device is a libfabric NIC
 * @return	Link speed index into `pcie_gen`, on success.
 * @return	Link width, on success.
 * @return	0, on sucess
 *		non-zero, on error
 */
static int get_pci_device_speed(hwloc_obj_t node, bool is_nic,
				size_t *speed_idx, size_t *width)
{
	struct hwloc_pcidev_attr_s attr;
	/* Override the following PCI width and speed of libfabric NICs with fallback values */
	char *override_width = "255";
	char *override_speed = "Unknown";
	size_t fallback_width = 8;
	size_t fallback_speed_idx = 3;

	if (node->type == HWLOC_OBJ_BRIDGE) {
		attr = node->attr->bridge.upstream.pci;
	} else if (node->type == HWLOC_OBJ_PCI_DEVICE) {
		attr = node->attr->pcidev;
	} else {
		NCCL_OFI_WARN("Expected topology node to be a PCI device or bridge");
		return ncclInternalError;
	}

	int ret;
	char prop_str[MAX_DEV_PROPERTY_LENGTH + 1];
	/* Size of the PCI speed lookup table `pcie_gen` */
	size_t num_pcie_gens = sizeof(pcie_gen) / sizeof(pcie_gen[0]);

	/* Read link speed */
	if ((ret = get_device_property(attr.domain, attr.bus, attr.dev, attr.func,
				       speed_name, prop_str))) {
		return ret;
	}

	/* Search reported PCI speed in `pcie_gen` lookup table */
	*speed_idx = 0;
	while (*speed_idx < num_pcie_gens && strncmp(prop_str, pcie_gen[*speed_idx], strlen(pcie_gen[*speed_idx])) != 0) {
		++(*speed_idx);
	}

	if (is_nic && strncmp(override_speed, prop_str, strlen(override_speed)) == 0) {
		/* Override speed */
		*speed_idx = fallback_speed_idx;
		NCCL_OFI_INFO(NCCL_INIT,
			      "Override link speed \"%s\" of NIC %04x:%02x:%02x.%01x with speed \"%s\"",
			      prop_str, attr.domain, attr.bus, attr.dev, attr.func, pcie_gen[*speed_idx]);
	}
	if (*speed_idx == num_pcie_gens) {
		NCCL_OFI_WARN("Unknown link speed \"%s\" of device %04x:%02x:%02x.%01x",
			      prop_str, attr.domain, attr.bus, attr.dev, attr.func);
		return -EINVAL;
	}

	/* Read link width */
	if ((ret = get_device_property(attr.domain, attr.bus, attr.dev, attr.func,
				       width_name, prop_str))) {
		return ret;
	}

	if (is_nic && strncmp(override_width, prop_str, strlen(override_width)) == 0) {
		/* Override width */
		*width = fallback_width;
		NCCL_OFI_INFO(NCCL_INIT,
			      "Override link width \"%s\" of NIC %04x:%02x:%02x.%01x with width \"%zu\"",
			      prop_str, attr.domain, attr.bus, attr.dev, attr.func, *width);
	} else {
		*width = strtol(prop_str, NULL, 0);
	}
	if (errno == ERANGE) {
		NCCL_OFI_WARN("Unable to convert link width \"%s\" of device %04x:%02x:%02x.%01x to a valid link width. Error: %s",
			      prop_str,
			      attr.domain, attr.bus, attr.dev, attr.func, strerror(errno));
		return -errno;
	} else if (*width == 0) {
		NCCL_OFI_WARN("Unknown link width \"%s\" of device %04x:%02x:%02x.%01x",
			      prop_str,
			      attr.domain, attr.bus, attr.dev, attr.func);
		return -EINVAL;
	}

	return 0;
}

/*
 * @brief	Read link speed and width of topology node and its parent topology node from file system and return minimum
 *
 * @param	node
 *		PCI device or bridge topology node. Parent of `node` must be a bridge topology node.
 * @param	is_nic
 *		True if device is a libfabric NIC
 * @return	Link speed index into `pci_gen`, on success.
 * @return	Link width, on success.
 * @return	0, on sucess
 *		non-zero, on error
 */
static int get_pci_device_min_speed(hwloc_obj_t node, bool is_nic, size_t *speed_idx,
				    size_t *width)
{
	int ret;
	hwloc_obj_t parent_node = node->parent;

	size_t device_width, port_width;
	size_t device_speed_idx, port_speed_idx;

	/* Read speed and width */
	if ((ret = get_pci_device_speed(node, is_nic, &device_speed_idx, &device_width))) {
		return ret;
	}

	/* Read speed and width of parent topology node to retrieve port speed and width. */
	if ((ret = get_pci_device_speed(parent_node, is_nic, &port_speed_idx, &port_width))) {
		return ret;
	}

	*speed_idx = NCCL_OFI_MIN(device_speed_idx, port_speed_idx);
	*width = NCCL_OFI_MIN(device_width, port_width);

	return 0;
}

/*
 * @brief	Write cpu opening tag to NCCL topology file
 *
 * @param	node
 *		Topology node of type `HWLOC_OBJ_NUMANODE`
 * @param	file
 *		Output file
 * @param	indent
 *		Indentation
 */
static int write_cpu_opening_tag(hwloc_obj_t node, FILE *file, int indent)
{

	/* Write NUMA node opening tag including `numaid` and
	 * `affinity`. Fields `arch`, `vendor`, `familyid`, and
	 * `modelid` are added by NCCL later. */
	if (fprintf(file,
		    "%*s"
		    "<cpu "
		    "numaid=\"%u\""
		    ">\n",
		    indent, "", node->os_index) < 0) {
		NCCL_OFI_WARN("Failed to print opening CPU tag. ERROR: %s",
			      strerror(errno));
		return -errno;
	}

	return 0;
}

/*
 * @brief	Write cpu closing tag to NCCL topology file
 *
 * @param	file
 *		Output file
 * @param	indent
 *		Indentation
 */
static int write_cpu_closing_tag(FILE *file, int indent)
{
	if (fprintf(file, "%*s</cpu>\n", indent, "") < 0) {
		NCCL_OFI_WARN("Failed to print closing CPU tag. ERROR: %s",
			      strerror(errno));
		return -errno;
	}

	return 0;
}

/*
 * @brief	Write pci tag to NCCL topology file
 *
 * @param	file
 *		Output file
 * @param	indent
 *		Indentation
 * @param	pcidev
 *		Pointer to pci device struct
 * @param	speed_idx
 *		Index into link speed lookup table `pcie_gen`
 * @param	width
 *		link width
 */
static int write_pci_tag(FILE *file, int indent,
			 struct hwloc_pcidev_attr_s *pcidev,
			 size_t speed_idx, size_t width)
{
	int rc = fprintf(file,
			 "%*s"
			 "<pci "
			 "busid=\"%04x:%02x:%02x.%01x\" "
			 "link_speed=\"%s GT/s PCIe/s\" "
			 "link_width=\"%ld\"/>\n",
			 indent, "",
			 pcidev->domain, pcidev->bus, pcidev->dev, pcidev->func,
			 pcie_gen[speed_idx],
			 width);

	if (rc < 0) {
		NCCL_OFI_WARN("Failed to print PCI tag. ERROR: %s",
			      strerror(errno));
		return -errno;
	}

	return 0;
}

/*
 * @brief	Write NIC pci tag to NCCL topology file
 *
 * @param	node
 *		Topology node of type ``HWLOC_OBJ_PCI_DEVICE`
 *		that stores a list of libfabric NIC info structs in its user data
 * @param	file
 *		Output file
 * @param	indent
 *		Indentation
 */
static int write_nic(hwloc_obj_t node, FILE *file, int indent)
{
	int ret = 0;
	size_t width;
	size_t speed_idx;
	nccl_ofi_topo_data_t *userdata = (nccl_ofi_topo_data_t *)node->userdata;
	int group_size = userdata->info_list_len;
	struct hwloc_pcidev_attr_s *pcidev = &node->attr->pcidev;

	/* Retrieve link speed and width of NIC */
	if ((ret = get_pci_device_min_speed(node, true, &speed_idx, &width))) {
		NCCL_OFI_WARN("Failed to retrieve PCI speed and width of NIC");
		return ret;
	}

	/* Scale NIC speed and width up to the speed of the GPU. Since
	 * NICs are grouped, GPU topology node is attached to the NIC
	 * topology node. */
	if (group_size > 1) {
		hwloc_obj_t gpu = userdata->gpu_group_node;
		size_t gpu_width;
		size_t gpu_speed_idx;
		assert(gpu);

		/* Retrieve link speed and width of GPU */
		if ((ret = get_pci_device_min_speed(gpu, false, &gpu_speed_idx, &gpu_width))) {
			NCCL_OFI_WARN("Failed to retrieve PCI speed and width of GPU associated to NIC");
			return ret;
		}

		/* In case we have multiple NICs in the group,
		 * increase link speed if possible and decrease NIC
		 * group size */
		while (group_size > 1 && speed_idx < gpu_speed_idx) {
			++speed_idx;
			group_size /= 2;
		}

		/* In case we still have multiple NICs in the group,
		 * increase link width if possible */
		while (group_size > 1 && 2 * width <= gpu_width) {
			width *= 2;
			group_size /= 2;
		}
	}

	if ((ret = write_pci_tag(file, indent, pcidev, speed_idx, width)) != 0) {
		NCCL_OFI_WARN("Failed to write PCI NIC tag");
	}

	return ret;
}

/*
 * @brief	Write pci opening tag to NCCL topology file
 *
 * @param	file
 *		Output file
 * @param	pcidev
 *		Pointer to pci device struct
 * @param	indent
 *		Indentation
 */
static int write_pci_opening_tag(FILE *file, struct hwloc_pcidev_attr_s *pcidev, int indent)
{
	int rc = fprintf(file,
			 "%*s"
			 "<pci "
			 "busid=\"%04x:%02x:%02x.%01x\">\n",
			 indent, "",
			 pcidev->domain, pcidev->bus, pcidev->dev, pcidev->func);
	if (rc < 0) {
		NCCL_OFI_WARN("Failed to print opening PCI tag. ERROR: %s",
			      strerror(errno));
		return -errno;
	}

	return 0;
}

/*
 * @brief	Write bridge pci opening tag to NCCL topology file
 *
 * @param	node
 *		Topology node of type `HWLOC_OBJ_BRIDGE`
 * @param	file
 *		Output file
 * @param	indent
 *		Indentation
 */
static int write_bridge_opening_tag(hwloc_obj_t node, FILE *file, int indent)
{
	struct hwloc_pcidev_attr_s *pcidev = &node->attr->bridge.upstream.pci;

	if (write_pci_opening_tag(file, pcidev, indent) < 0) {
		NCCL_OFI_WARN("Failed to print opening PCI bridge tag");
		return -errno;
	}

	return 0;
}

/*
 * @brief	Write pci closing tag to NCCL topology file
 *
 * @param	file
 *		Output file
 * @param	indent
 *		Indentation
 */
static int write_pci_closing_tag(FILE *file, int indent)
{
	if (fprintf(file, "%*s</pci>\n", indent, "") < 0) {
		NCCL_OFI_WARN("Failed to write closing PCI bridge tag. ERROR: %s",
			      strerror(errno));
		return -errno;
	}

	return 0;
}

/*
 * @brief	Helper function to write NCCL topology file recursively based on NCCL OFI topology
 *
 * @param	topo
 *		Hwloc topology
 * @param	node
 *		Current hwloc topology node
 * @param	file
 *		The file to print to
 * @param	indent
 * @param	Number of bridges upstream of this node
 *		Indicate current indentation level
 * @param	bridge_depth
 *		Number of PCI bridges upstream of this topology node
 * @return	0, on success
 *		non-zero, on error
 */
static int write_nccl_topo_rec(hwloc_topology_t topo, hwloc_obj_t node, FILE *file, int indent, int bridge_depth)
{
	int ret = 0;
	int indent_offset = 2;
	bool close_numanode = false;
	bool close_bridge = false;
	hwloc_obj_t numa_mem_child = NULL;
	hwloc_obj_t child = NULL;
	nccl_ofi_topo_data_t *topo_data = (nccl_ofi_topo_data_t *)node->userdata;

	/* Only nodes with NICs or Nvidia GPUs in its subtree store
	 * store userdata. Use this information to avoid printing
	 * parts of the topology without NICs and Nvidia GPUs. */
	if (!topo_data) return ret;

	if (node->type == HWLOC_OBJ_BRIDGE) {
		if (!node->attr) {
			NCCL_OFI_WARN("Bridge is missing attribute struct");
			return -EINVAL;
		}

		/* Do not print Host PCIe switch, represented by the
		 * two PCI bridges on depth 0 and 1. Print remaining
		 * PCIe switches, represented by two devices each */
		if (bridge_depth >= 2 && bridge_depth % 2 == 0) {
			if ((ret = write_bridge_opening_tag(node, file, indent))) {
				return ret;
			}
			close_bridge = true;
			indent += indent_offset;
		}

		++bridge_depth;
	} else if (node->type == HWLOC_OBJ_PCI_DEVICE &&
		   topo_data->info_list) {
			/* Topology nodes which store NIC info lists
			 * are topology nodes of leader NICs. The
			 * leader NIC is the first NIC in the list. */
			if ((ret = write_nic(node, file, indent))) {
				return ret;
			}
			indent += indent_offset;
	} else if (node->type == HWLOC_OBJ_NUMANODE) {
		/* Before HWLOC 2.0, NUMA topology nodes are stored in
		 * the normal topology tree */
		if ((ret = write_cpu_opening_tag(node, file, indent))) {
			return ret;
		}
		close_numanode = true;
		indent += indent_offset;
	} else if ((numa_mem_child = get_numa_mem_child(node))) {
		/* HWLOC 2.0 moved NUMA nodes from the normal topology
		 * tree to list of memory children. The consequence is
		 * that NUMA nodes may "float" in a package, and thus,
		 * no NUMA node will be found on the path from the
		 * root to a PCI device that needs to be
		 * written. Collect those NUMA nodes and write
		 * them. However, in case a NUMA node is found in the memory
		 * children list and the NUMA node stores user data,
		 * the NUMA node is on the path to a PCI device and
		 * will be printed in the next recursion. */
		if (!numa_mem_child->userdata) {
			if ((ret = write_cpu_opening_tag(numa_mem_child, file, indent))) {
				return ret;
			}
			close_numanode = true;
			indent += indent_offset;
		}
	}

	/* Recurse */
	while ((child = hwloc_get_next_child(topo, node, child))) {
		if ((ret = write_nccl_topo_rec(topo, child,
					       file, indent,
					       bridge_depth))) {
			return ret;
		}
	}

	if (close_numanode) ret = write_cpu_closing_tag(file, indent - indent_offset);
	else if (close_bridge) ret = write_pci_closing_tag(file, indent - indent_offset);

	return ret;
}

int nccl_ofi_topo_write(nccl_ofi_topo_t *topo, FILE *file)
{
	int ret = 0;
	int bridge_depth = 0;
	int indent = 2;

	if (fprintf(file, "<system version=\"1\">\n") < 0) {
		NCCL_OFI_WARN("Failed to write topology to NCCL topology file. ERROR: %s",
			      strerror(errno));
		return -errno;
	}

	ret = write_nccl_topo_rec(topo->topo, hwloc_get_root_obj(topo->topo),
				   file, indent, bridge_depth);
	if (ret) {
		NCCL_OFI_WARN("Failed to write topology to NCCL topology file");
		return ret;
	}

	if (fprintf(file, "</system>") < 0) {
		NCCL_OFI_WARN("Failed to write topology to NCCL topology file. ERROR: %s",
			      strerror(errno));
		return -errno;
	}

	return ret;
}

int nccl_ofi_topo_num_info_lists(nccl_ofi_topo_t *topo, int *num_lists)
{
	if (!topo || !topo->data_vec) {
		NCCL_OFI_WARN("Invalid topology. Topology is not initialized.");
		return -EINVAL;
	}

	nccl_ofi_topo_data_t *data = NULL;
	nccl_ofi_topo_data_iterator_t data_iter;
	nccl_ofi_topo_set_to_begin(topo, &data_iter);

	*num_lists = 0;
	while ((data = nccl_ofi_get_user_data(&data_iter))) {
		nccl_ofi_inc_user_data_iter(&data_iter);
		*num_lists += (!data->info_list) ? 0 : 1;
	}

	return 0;
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
