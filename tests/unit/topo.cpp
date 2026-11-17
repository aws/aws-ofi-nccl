/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdint.h>

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

static bool check_group_scaling(const char *name, int group_size,
				size_t speed_idx, size_t width,
				size_t max_speed_idx, size_t max_width,
				int expected_group_size,
				size_t expected_speed_idx,
				size_t expected_width)
{
	nccl_ofi_topo_scale_group(group_size, speed_idx, width,
				  max_speed_idx, max_width);
	if (group_size != expected_group_size || speed_idx != expected_speed_idx ||
	    width != expected_width) {
		fprintf(stderr,
			"%s: got group %d, speed index %zu, width %zu; "
			"expected %d, %zu, %zu\n",
			name, group_size, speed_idx, width, expected_group_size,
			expected_speed_idx, expected_width);
		return false;
	}
	return true;
}

static bool test_group_scaling()
{
	/* A two-NIC group already at the maximum speed uses width to
	 * represent its second NIC. */
	if (!check_group_scaling("maximum speed", 2, 5, 16, 5, SIZE_MAX,
				 1, 5, 32)) {
		return false;
	}

	/* A four-NIC group uses one speed increase and one width increase. */
	if (!check_group_scaling("speed and width", 4, 4, 16, 5, SIZE_MAX,
				 1, 5, 32)) {
		return false;
	}

	/* An eight-NIC group reaches the speed limit through three increases. */
	if (!check_group_scaling("speed only", 8, 2, 16, 5, SIZE_MAX,
				 1, 5, 16)) {
		return false;
	}

	/* A finite width limit can leave part of the group unrepresented. */
	if (!check_group_scaling("limited width", 8, 4, 16, 5, 32,
				 2, 5, 32)) {
		return false;
	}

	/* The width guard must not overflow when no further doubling fits. */
	constexpr size_t large_width = SIZE_MAX / 2 + 1;
	if (!check_group_scaling("width overflow", 2, 5, large_width, 5, SIZE_MAX,
				 2, 5, large_width)) {
		return false;
	}

	return true;
}

int main()
{
	unit_test_init();

	hwloc_obj root = make_obj(HWLOC_OBJ_MACHINE, 0, nullptr);
	hwloc_obj package = make_obj(HWLOC_OBJ_PACKAGE, 1, &root);

	union hwloc_obj_attr_u host_attr = {};
	hwloc_obj host = make_bridge(HWLOC_OBJ_BRIDGE_HOST, &package, &host_attr);

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
	hwloc_obj other_host = make_bridge(HWLOC_OBJ_BRIDGE_HOST, &package, &other_host_attr);
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

	/* Count real bridge parents rather than hwloc's virtual object depth. */
	hwloc_obj_t found_package = nullptr;
	unsigned int bridge_depth = 0;
	if (!nccl_ofi_topo_get_package_bridge_depth(&nic, &found_package, &bridge_depth) ||
	    found_package != &package || bridge_depth != 2) {
		fprintf(stderr, "expected NIC bridge depth 2\n");
		return 1;
	}
	if (!nccl_ofi_topo_get_package_bridge_depth(
		    &nested_gpu, &found_package, &bridge_depth) ||
	    found_package != &package || bridge_depth != 3) {
		fprintf(stderr, "expected nested GPU bridge depth 3\n");
		return 1;
	}
	if (nccl_ofi_topo_get_package_bridge_depth(
		    &root, &found_package, &bridge_depth)) {
		fprintf(stderr, "machine object must not have a package-relative depth\n");
		return 1;
	}

	/* Find the NIC tier matching the common accelerator bridge count. */
	const unsigned int accel_depths[] = {3, 3, 3, 3};
	const unsigned int mixed_nic_depths[] = {3, 3, 3, 3, 5, 5};
	unsigned int matching_nic_depth = 0;
	if (!nccl_ofi_topo_get_accel_matching_nic_bridge_depth(
		    accel_depths, 4, mixed_nic_depths, 6, &matching_nic_depth) ||
	    matching_nic_depth != 3) {
		fprintf(stderr, "expected unambiguous matching NIC depth 3\n");
		return 1;
	}

	const unsigned int uniform_nic_depths[] = {3, 3};
	const unsigned int multiple_accel_depths[] = {3, 3, 3, 4};
	const unsigned int nonmatching_nic_depths[] = {4, 5};
	if (nccl_ofi_topo_get_accel_matching_nic_bridge_depth(
		    accel_depths, 4, uniform_nic_depths, 2, &matching_nic_depth) ||
	    nccl_ofi_topo_get_accel_matching_nic_bridge_depth(
		    multiple_accel_depths, 4, mixed_nic_depths, 6,
		    &matching_nic_depth) ||
	    nccl_ofi_topo_get_accel_matching_nic_bridge_depth(
		    accel_depths, 4, nonmatching_nic_depths, 2,
		    &matching_nic_depth)) {
		fprintf(stderr, "ambiguous package depth selection must fail open\n");
		return 1;
	}

	if (!test_group_scaling()) {
		return 1;
	}

	printf("Topology tests completed successfully!\n");
	return 0;
}
