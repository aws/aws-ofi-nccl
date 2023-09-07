/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_NET_OFI_TOPO_H_
#define NCCL_NET_OFI_TOPO_H_

#ifdef _cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <hwloc.h>
#include <rdma/fabric.h>

#include "nccl-headers/error.h"

/*
 * @brief	User data of topology nodes
 *
 * User data is used to add data to hardware topology nodes. The user
 * data is used by NCCL OFI topology and its associated functions.
 *
 * User data stores members that are temporarily used by libfabric NIC
 * info grouping functions nccl_ofi_topo_create() and nccl_ofi_topo_group().
 *
 * The 'info_list' member of user data is used to store Libfabric NIC
 * info structs in topology nodes: After loading a NCCL OFI topology,
 * libfabric NIC info structs are stored in their corresponding
 * topology nodes. After the grouping functions are executed, the
 * libfabric NIC info lists (aka groups) are stored in the topology
 * nodes of the group leaders (first NIC info of the list).
 */
typedef struct nccl_ofi_topo_data {
	/* Libfabric NIC info list */
	struct fi_info *info_list;

	/* Length of libfabric NIC info list */
	int info_list_len;

	/* Temporary data used by grouping algorithms. This member
	 * tracks the number of groups that are supposed to be created
	 * from the libfabric NIC info list stored in this struct
	 * while grouping algorithm is executed. */
	int num_groups;

	/* One of the GPU topology nodes that is the closest to the
	 * NICs in `info_list` */
	hwloc_obj_t gpu_group_node;

	/* Temporary data for grouping algorithm. Indicates whether
	 * the corresponding topology node of this object has
	 * libfabric NIC info lists stored in its subtree. */
	bool is_nic_subtree;

	/* Track whether the corresponding topology node has been
	 * detected as a GPU and contributed to the group count
	 * already */
	bool contributed_gpu;

	/* Backward pointer to corresponding topology node */
	hwloc_obj_t node;
} nccl_ofi_topo_data_t;

/*
 * @brief	Vector of nccl_ofi_topo_data_t structs
 */
typedef struct nccl_ofi_topo_data_vec {
	/* Array of user data structs. NULL if size is zero. */
	nccl_ofi_topo_data_t *data;

	/* Size of vector */
	size_t size;
} nccl_ofi_topo_data_vec_t;

/*
 * @brief	nccl_ofi_topo_data_vec_t forward iterator
 */
typedef struct nccl_ofi_topo_data_iterator {
	/* Current position of the iterator */
	nccl_ofi_topo_data_t *begin;

	/* End of iterator */
	nccl_ofi_topo_data_t *end;
} nccl_ofi_topo_data_iterator_t;

/*
 * @brief	NCCL OFI topology containing hardware topology and topology node user data
 */
typedef struct nccl_ofi_topo {
	/* Hardware topology. Each topology node stores a pointer to a
	 * different object of vector 'data_vec'. */
	hwloc_topology_t topo;

	/* Maximum number of libfabric NICs in a group */
	int max_group_size;

	/* Vector of topology node user data. The user data objects of
	 * the vector are the vehicle to store temporary data as well
	 * as NIC info lists in the topology nodes. There is a
	 * one-to-one relationship between each topology node of
	 * 'topo' and user data objects of this vector. */
	nccl_ofi_topo_data_vec_t *data_vec;
} nccl_ofi_topo_t;

/*
 * @brief	Set topology user data iterator to the first element of NCCL OFI
 *		topologies's user data array
 */
int nccl_ofi_topo_set_to_begin(nccl_ofi_topo_t *topo,
			       nccl_ofi_topo_data_iterator_t *iter);

/*
 * @brief 	Free NCCL OFI topology
 *
 * Free NCCL OFI topology including its hardware topology, hardware
 * topology user data, and the libfabric NIC info lists stored in the
 * user data. Operation is only executed if input topology is not NULL.
 */
void nccl_ofi_topo_free(nccl_ofi_topo_t *topo);

/*
 * @brief 	Group libfabric NIC info structs according to locality to GPU topology nodes
 *
 * The grouping algorithm performs the following steps:
 *
 * 1. For each topology node that stores a libfabric NIC info list in its
 * user data, mark the node and its ancestors.
 * Note: Now, all marked nodes have at least one NIC info object in its
 * subtree.
 * 2. Use function `propagate_gpu_count()` to propagate GPU counts to
 * marked topology nodes: For each Nvidia GPU topology node, walk up
 * the tree and search for a marked node. If a marked node is found,
 * increase the node's 'num_groups' by one (initialized by zero).
 * 3. For each topology node with a libfabric NIC info list in its
 * user data, search towards the root for a node with 'num_groups' > 0.
 * If such a node has been found, move the list to that node.
 * 4. For each topology node with a libfabric NIC info list and 'num_groups > 0',
 * split the list into 'num_groups' sub-lists (groups) and add the
 * groups to the topology nodes corresponding to their leaders (first
 * NIC info of the list). If the length of the list is not a
 * multiple of 'num_groups', distribute the members evenly to the
 * groups with the exception that the first groups get assigned an
 * additional member.
 *
 * 
 * The example below shows a schematic representation of the topology
 * while executing the grouping algorithm. The topology shows one
 * bridge, two nodes that correspond to a Nvidia GPUs and four nodes
 * that correspond to libfabric NIC info objects.
 * Note that the schematic view below is a simplification of an actual
 * hardware topology. It is also only a selective view of a
 * hypothetical hardware topology.
 * 
 * Legend:
 * Topology nodes:
 * X:YZ
 *  |
 * 
 * Topology node type X:
 * B: Bridge
 * G: GPU
 * N: Node corresponding to a NIC
 * 
 * num_groups Y:
 * Number indicating the group count
 * 
 * Mark Z:
 * u: Unmarked node
 * m: Marked node
 * 
 * Pointer "|":
 * The symbol "|" indicates the libfabric NIC info list pointer stored in
 * the user data of the topology node. The list pointer may be NULL (no
 * list) or stores a list of info objects (NIC0, NIC1, NIC2, NIC3, NIC4).
 * 
 * 
 * Initial state after NCCL OFI topology has been created via function
 * nccl_ofi_topo_create().
 * 
 *                            /
 *                        B:0u
 *       /        /    |        |       \        \
 *     /        /      |        |        \        \
 *   /        /        |        |         \        \
 * G:0u     G:0u     N:0u     N:0u        N:0u    N:0u
 *  |        |        |        |           |        |
 *                    NIC0     NIC1        NIC2     NIC3
 * 
 * State after step 1. Topology nodes with libfabric NIC info structs and
 * the nodes' ancestors are marked.
 * 
 *                            /
 *                        B:0m
 *       /        /   |        |       \        \
 *     /        /     |        |        \        \
 *   /         /      |        |         \        \
 * G:0u     G:0u     N:0m     N:0m        N:0m    N:0m
 *  |        |        |        |           |        |
 *                    NIC0     NIC1        NIC2     NIC3
 * 
 * State after step 2. 'num_groups' of bridge node (closest marked node of
 * GPU topology nodes) is increased to two.
 * 
 *                            /
 *                        B:2m
 *       /        /   |        |       \        \
 *     /        /     |        |        \        \
 *   /         /      |        |         \        \
 * G:0u     G:0u     N:0m     N:0m        N:0m    N:0m
 *  |        |        |        |           |        |
 *                    NIC0     NIC1        NIC2     NIC3
 * 
 * State after step 3. The libfabric NIC info structs are lifted up to
 * bridge node (node with 'num_groups' larger then zero).
 * 
 *                             NIC0-NIC1-NIC2-NIC3
 *                            /
 *                        B:2m
 *       /        /   |        |       \        \
 *     /        /     |        |        \        \
 *   /         /      |        |         \        \
 * G:0u     G:0u     N:0m     N:0m        N:0m    N:0m
 *  |        |        |        |           |        |
 * 
 * State after step 4. List of libfabric NIC info structs is split into
 * two groups. Each group is attached to the topology node of the head of
 * the list.
 * 
 *                            /
 *                        B:0m
 *       /        /   |        |       \        \
 *     /        /     |        |        \        \
 *   /         /      |        |         \        \
 * G:0u     G:0u     N:0m     N:0m        N:0m    N:0m
 *  |        |        |        |           |        |
 *                    NIC0                 NIC2
 *                    |                    |
 *                    NIC1                 NIC3
 * 
 * @param	topo
 *		The NCCL OFI topology.
 *
 * @return	ncclSuccess, on success
 * 		ncclInvalidArgument, if unable to extract libfabric
 * 		NIC info from topology via bus id
 * 		ncclInternalError, on others
 */
int nccl_ofi_topo_group(nccl_ofi_topo_t *topo);

/*
 * @brief	Allocate and initialize nccl_ofi_topo_t struct
 *
 * Create a nccl_ofi_topo_t struct that stores the hardware topology
 * of the machine and add libfabric NIC info structs to their
 * corresponding topology nodes. Note that this function duplicates
 * the info structs.
 *
 * @param	info_list
 *		List of libfabric NIC info structs
 * @return	NCCL OFI hardware topology, on success
 *		NULL, on others
 */
nccl_ofi_topo_t *nccl_ofi_topo_create(struct fi_info *info_list);

/*
 * @brief	Write NCCL topology file based on NCCL OFI topology
 *
 * @param	topo
 *		NCCL OFI topology
 * @param	file
 *		File to write to
 * @return	0, on success
 *		non-zero, on error
 */
int nccl_ofi_topo_write(nccl_ofi_topo_t *topo, FILE *file);

/*
 * @brief	Return number of topology nodes that store a libfabric NIC info
 *		list
 *
 * @param	topo
 *		The NCCL OFI topology
 *
 * @return	Number of lists, on success
 * 		undefined, on others
 * @return	ncclInvalidArgument, if topology is not initialized
 *		ncclSuccess, on success
 *
 */
int nccl_ofi_topo_num_info_lists(nccl_ofi_topo_t *topo, int *num_lists);

/*
 * @brief	Return next libfabric NIC info list from NCCL OFI topology
 *
 * This function iterates over the user data of the NCCL OFI topology
 * and returns the next libfabric NIC info list.
 *
 * @param	topo
 *		NCCL OFI topology
 * @return	next libfabric NIC info list, if available
 *		NULL, if end of vector is reached and no list has been found
 */
struct fi_info *nccl_ofi_topo_next_info_list(nccl_ofi_topo_data_iterator_t *iter);

/*
 * @brief	Dump NCCL topology into file
 *
 * @param	topo
 * 		The topology
 * @param	file
 *		The file
 * @return	0, on success
 *		non-zero, on error
 */
int nccl_ofi_topo_write_nccl_topology(nccl_ofi_topo_t *topo, FILE *file);

#ifdef _cplusplus
}
#endif

#endif // End NCCL_NET_OFI_TOPO_H_
