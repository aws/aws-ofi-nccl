/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>

#include "unit_test.h"
#include "nccl_ofi_topo.h"

static hwloc_obj make_obj(hwloc_obj_type_t type, int depth, hwloc_obj_t parent,
			  union hwloc_obj_attr_u *attr = nullptr)
{
	hwloc_obj obj = {};
	obj.type = type;
	obj.depth = depth;
	obj.parent = parent;
	obj.attr = attr;
	return obj;
}

static hwloc_obj make_bridge(hwloc_obj_bridge_type_t upstream, hwloc_obj_t parent,
			     union hwloc_obj_attr_u *attr)
{
	attr->bridge.upstream_type = upstream;
	return make_obj(HWLOC_OBJ_BRIDGE, HWLOC_TYPE_DEPTH_BRIDGE, parent, attr);
}

static hwloc_obj make_pcidev(hwloc_obj_t parent)
{
	return make_obj(HWLOC_OBJ_PCI_DEVICE, HWLOC_TYPE_DEPTH_PCI_DEVICE, parent);
}

static bool check(const char *name, bool actual, bool expected)
{
	if (actual != expected) {
		fprintf(stderr, "%s: expected %s\n", name, expected ? "true" : "false");
		return false;
	}
	return true;
}

/*
 * Scale a NIC group and compare the result with what is expected.
 *
 * group_size is the number of NICs in the group that the scaling to the
 * GPU's PCI link has not yet represented. width is the synthetic link width
 * of the leader NIC, in PCI lanes. 16 is a typical x16 link.
 */
static bool check_scaled(const char *name, bool is_coherent, int group_size, size_t width,
			 int expected_group_size, size_t expected_width)
{
	int actual_group_size = group_size;
	size_t actual_width = width;

	nccl_ofi_topo_scale_coherent_group(is_coherent, actual_group_size, actual_width);

	if (actual_group_size != expected_group_size || actual_width != expected_width) {
		fprintf(stderr, "%s: got group size %d, width %zu; expected %d, %zu\n", name,
			actual_group_size, actual_width, expected_group_size, expected_width);
		return false;
	}
	return true;
}

static bool test_scale_coherent_group()
{
	/* Coherent GPU, two NICs, x16 leader: the second NIC is represented by
	 * doubling the width to x32, which leaves no NIC unrepresented. */
	if (!check_scaled("coherent two-NIC group", true, 2, 16, 1, 32)) {
		return false;
	}

	/* Non-coherent GPU: its PCI link still bounds the group, so nothing
	 * changes and both NICs stay unrepresented. */
	if (!check_scaled("non-coherent two-NIC group", false, 2, 16, 2, 16)) {
		return false;
	}

	/* Coherent GPU, four NICs: two doublings represent all four, so the
	 * width goes x16 -> x32 -> x64. */
	if (!check_scaled("coherent four-NIC group", true, 4, 16, 1, 64)) {
		return false;
	}

	/* Coherent GPU, eight NICs: three doublings, x16 -> x128. */
	if (!check_scaled("coherent eight-NIC group", true, 8, 16, 1, 128)) {
		return false;
	}

	/* Non-coherent GPU, four NICs: nothing changes for any group size. */
	if (!check_scaled("non-coherent four-NIC group", false, 4, 16, 4, 16)) {
		return false;
	}

	/* Coherent GPU, one NIC: nothing is left to represent, so the width
	 * must not be doubled. */
	if (!check_scaled("coherent single NIC", true, 1, 16, 1, 16)) {
		return false;
	}

	return true;
}

int main()
{
	unit_test_init();

	hwloc_obj root = make_obj(HWLOC_OBJ_MACHINE, 0, nullptr);

	union hwloc_obj_attr_u host_attr = {};
	hwloc_obj host = make_bridge(HWLOC_OBJ_BRIDGE_HOST, &root, &host_attr);

	union hwloc_obj_attr_u switch_attr = {};
	hwloc_obj pcie_switch = make_bridge(HWLOC_OBJ_BRIDGE_PCI, &host, &switch_attr);

	/* NIC and GPU below the same PCIe switch: the PCIe path exists. */
	hwloc_obj nic = make_pcidev(&pcie_switch);
	hwloc_obj gpu = make_pcidev(&pcie_switch);
	if (!check("shared PCIe switch",
		   nccl_ofi_topo_share_pcie_switch(&nic, &gpu), true)) {
		return 1;
	}

	/* A host bridge is not a PCIe switch, even as the closest common parent. */
	hwloc_obj host_nic = make_pcidev(&host);
	hwloc_obj host_gpu = make_pcidev(&host);
	if (!check("shared host bridge",
		   nccl_ofi_topo_share_pcie_switch(&host_nic, &host_gpu), false)) {
		return 1;
	}

	/* hwloc gives I/O objects virtual depths by type, so branches holding
	 * different numbers of bridges must still resolve through parent links. */
	if (!check("asymmetric branch, no shared switch",
		   nccl_ofi_topo_share_pcie_switch(&host_nic, &gpu), false)) {
		return 1;
	}

	union hwloc_obj_attr_u nested_attr = {};
	hwloc_obj nested_switch = make_bridge(HWLOC_OBJ_BRIDGE_PCI, &pcie_switch, &nested_attr);
	hwloc_obj nested_gpu = make_pcidev(&nested_switch);
	if (!check("asymmetric branch, shared switch",
		   nccl_ofi_topo_share_pcie_switch(&nic, &nested_gpu), true)) {
		return 1;
	}

	/* Separate host bridges never share a PCIe switch. */
	union hwloc_obj_attr_u other_host_attr = {};
	hwloc_obj other_host = make_bridge(HWLOC_OBJ_BRIDGE_HOST, &root, &other_host_attr);
	hwloc_obj other_gpu = make_pcidev(&other_host);
	if (!check("disjoint host bridges",
		   nccl_ofi_topo_share_pcie_switch(&host_nic, &other_gpu), false)) {
		return 1;
	}

	/* A node topology could not place is not evidence of a PCIe path. */
	if (!check("missing GPU", nccl_ofi_topo_share_pcie_switch(&nic, nullptr), false) ||
	    !check("missing NIC", nccl_ofi_topo_share_pcie_switch(nullptr, &gpu), false)) {
		return 1;
	}

	if (!test_scale_coherent_group()) {
		return 1;
	}

	printf("Topology tests completed successfully!\n");
	return 0;
}
