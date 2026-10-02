/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <rdma/fabric.h>

#include "nccl_ofi_topo.h"
#include "topo_fixture.h"

namespace {

/*
 * Static fixture data.
 *
 * Platform expectations are typed arrays in this file. Each NIC entry carries
 * its numeric PCI BDF and expected nic_gpu_share_pcie_switch() result.
 * Scenarios select contiguous slices of NIC entries. A platform fixture names
 * a sanitized hwloc XML and its scenarios. The runner consumes these
 * generically, so adding a platform requires its arrays, topology.xml, and one
 * fixture_registry entry. No runner branch is platform specific.
 */

/* One NIC expectation: a numeric PCI BDF and the expected query result for the
 * group this NIC leads. The interface name is kept only for diagnostics;
 * grouping and matching use the numeric BDF fields directly. */
struct nic_expectation {
	const char *nic;	/* diagnostic only; not matched */
	uint16_t domain;
	uint8_t bus;
	uint8_t device;
	uint8_t function;
	bool expected_share_pcie_switch;
};

/* A named scenario: a contiguous slice of a platform's NIC expectation array
 * that is populated and grouped together as one fi_info list. */
struct scenario {
	const char *name;
	const nic_expectation *nics;
	size_t num_nics;
};

/* One platform fixture: a name, its XML path relative to the fixture root, and
 * its scenarios. */
struct platform_fixture {
	const char *platform;
	const char *xml_relpath;
	const scenario *scenarios;
	size_t num_scenarios;
};

/*
 * p6e-gb200 NIC expectations.
 *
 * A NIC expects true when its closest common ancestor with the associated GPU
 * is a PCI-to-PCI bridge; otherwise it expects false. The GPU-local NICs below
 * share two PCIe bridges with their GPU, while the CPU-mediated NICs share no
 * bridge with their GPU.
 *
 * Keep each class contiguous so PCIe-only and C2C-only are slices of this
 * canonical array, while mixed consumes the entire array.
 */
const nic_expectation p6e_gb200_nics[] = {
	/* GPU-local PCIe NICs */
	{ "rdmap39s0",  0x0000, 0x27, 0x00, 0x0, true },
	{ "rdmap40s0",  0x0000, 0x28, 0x00, 0x0, true },
	{ "rdmap61s0",  0x0000, 0x3d, 0x00, 0x0, true },
	{ "rdmap62s0",  0x0000, 0x3e, 0x00, 0x0, true },
	{ "rdmap154s0", 0x0000, 0x9a, 0x00, 0x0, true },
	{ "rdmap155s0", 0x0000, 0x9b, 0x00, 0x0, true },
	{ "rdmap176s0", 0x0000, 0xb0, 0x00, 0x0, true },
	{ "rdmap177s0", 0x0000, 0xb1, 0x00, 0x0, true },

	/* CPU-mediated C2C NICs */
	{ "rdmap54s0",  0x0000, 0x36, 0x00, 0x0, false },
	{ "rdmap55s0",  0x0000, 0x37, 0x00, 0x0, false },
	{ "rdmap76s0",  0x0000, 0x4c, 0x00, 0x0, false },
	{ "rdmap77s0",  0x0000, 0x4d, 0x00, 0x0, false },
	{ "rdmap169s0", 0x0000, 0xa9, 0x00, 0x0, false },
	{ "rdmap170s0", 0x0000, 0xaa, 0x00, 0x0, false },
	{ "rdmap191s0", 0x0000, 0xbf, 0x00, 0x0, false },
	{ "rdmap192s0", 0x0000, 0xc0, 0x00, 0x0, false },
};

constexpr size_t p6e_gb200_pcie_count = 8;
constexpr size_t p6e_gb200_c2c_count = 8;
constexpr size_t p6e_gb200_nic_count =
	sizeof(p6e_gb200_nics) / sizeof(p6e_gb200_nics[0]);
static_assert(p6e_gb200_pcie_count + p6e_gb200_c2c_count ==
	      p6e_gb200_nic_count);

const scenario p6e_gb200_scenarios[] = {
	{ "pcie-only", p6e_gb200_nics, p6e_gb200_pcie_count },
	{ "c2c-only", p6e_gb200_nics + p6e_gb200_pcie_count,
	  p6e_gb200_c2c_count },
	{ "mixed", p6e_gb200_nics, p6e_gb200_nic_count },
};

/* p5en places all sixteen EFA NICs behind PCIe switches shared with GPUs. */
const nic_expectation p5en_nics[] = {
	{ "rdmap85s0",  0x0000, 0x55, 0x00, 0x0, true },
	{ "rdmap86s0",  0x0000, 0x56, 0x00, 0x0, true },
	{ "rdmap87s0",  0x0000, 0x57, 0x00, 0x0, true },
	{ "rdmap88s0",  0x0000, 0x58, 0x00, 0x0, true },
	{ "rdmap110s0", 0x0000, 0x6e, 0x00, 0x0, true },
	{ "rdmap111s0", 0x0000, 0x6f, 0x00, 0x0, true },
	{ "rdmap112s0", 0x0000, 0x70, 0x00, 0x0, true },
	{ "rdmap113s0", 0x0000, 0x71, 0x00, 0x0, true },
	{ "rdmap135s0", 0x0000, 0x87, 0x00, 0x0, true },
	{ "rdmap136s0", 0x0000, 0x88, 0x00, 0x0, true },
	{ "rdmap137s0", 0x0000, 0x89, 0x00, 0x0, true },
	{ "rdmap138s0", 0x0000, 0x8a, 0x00, 0x0, true },
	{ "rdmap160s0", 0x0000, 0xa0, 0x00, 0x0, true },
	{ "rdmap161s0", 0x0000, 0xa1, 0x00, 0x0, true },
	{ "rdmap162s0", 0x0000, 0xa2, 0x00, 0x0, true },
	{ "rdmap163s0", 0x0000, 0xa3, 0x00, 0x0, true },
};

constexpr size_t p5en_nic_count = sizeof(p5en_nics) / sizeof(p5en_nics[0]);
static_assert(p5en_nic_count == 16);

const scenario p5en_scenarios[] = {
	{ "all", p5en_nics, p5en_nic_count },
};

/* p4d EFA NICs and GPUs meet at host bridges, not PCIe switches. */
const nic_expectation p4d_nics[] = {
	{ "rdmap16s27",  0x0000, 0x10, 0x1b, 0x0, false },
	{ "rdmap32s27",  0x0000, 0x20, 0x1b, 0x0, false },
	{ "rdmap144s27", 0x0000, 0x90, 0x1b, 0x0, false },
	{ "rdmap160s27", 0x0000, 0xa0, 0x1b, 0x0, false },
};

constexpr size_t p4d_nic_count = sizeof(p4d_nics) / sizeof(p4d_nics[0]);
static_assert(p4d_nic_count == 4);

const scenario p4d_scenarios[] = {
	{ "all", p4d_nics, p4d_nic_count },
};

/* p6-b200 places its eight Amazon EFA NICs behind shared PCIe switches. */
const nic_expectation p6_b200_nics[] = {
	{ "rdmap79s0",  0x0000, 0x4f, 0x00, 0x0, true },
	{ "rdmap80s0",  0x0000, 0x50, 0x00, 0x0, true },
	{ "rdmap96s0",  0x0000, 0x60, 0x00, 0x0, true },
	{ "rdmap97s0",  0x0000, 0x61, 0x00, 0x0, true },
	{ "rdmap113s0", 0x0000, 0x71, 0x00, 0x0, true },
	{ "rdmap114s0", 0x0000, 0x72, 0x00, 0x0, true },
	{ "rdmap132s0", 0x0000, 0x84, 0x00, 0x0, true },
	{ "rdmap133s0", 0x0000, 0x85, 0x00, 0x0, true },
};

constexpr size_t p6_b200_nic_count =
	sizeof(p6_b200_nics) / sizeof(p6_b200_nics[0]);
static_assert(p6_b200_nic_count == 8);

const scenario p6_b200_scenarios[] = {
	{ "all", p6_b200_nics, p6_b200_nic_count },
};

/* p6-b300 places its sixteen Amazon EFA NICs behind shared PCIe switches. */
const nic_expectation p6_b300_nics[] = {
	{ "rdmap86s0",  0x0000, 0x56, 0x00, 0x0, true },
	{ "rdmap87s0",  0x0000, 0x57, 0x00, 0x0, true },
	{ "rdmap101s0", 0x0000, 0x65, 0x00, 0x0, true },
	{ "rdmap102s0", 0x0000, 0x66, 0x00, 0x0, true },
	{ "rdmap116s0", 0x0000, 0x74, 0x00, 0x0, true },
	{ "rdmap117s0", 0x0000, 0x75, 0x00, 0x0, true },
	{ "rdmap131s0", 0x0000, 0x83, 0x00, 0x0, true },
	{ "rdmap132s0", 0x0000, 0x84, 0x00, 0x0, true },
	{ "rdmap146s0", 0x0000, 0x92, 0x00, 0x0, true },
	{ "rdmap147s0", 0x0000, 0x93, 0x00, 0x0, true },
	{ "rdmap161s0", 0x0000, 0xa1, 0x00, 0x0, true },
	{ "rdmap162s0", 0x0000, 0xa2, 0x00, 0x0, true },
	{ "rdmap176s0", 0x0000, 0xb0, 0x00, 0x0, true },
	{ "rdmap177s0", 0x0000, 0xb1, 0x00, 0x0, true },
	{ "rdmap191s0", 0x0000, 0xbf, 0x00, 0x0, true },
	{ "rdmap192s0", 0x0000, 0xc0, 0x00, 0x0, true },
};

constexpr size_t p6_b300_nic_count =
	sizeof(p6_b300_nics) / sizeof(p6_b300_nics[0]);
static_assert(p6_b300_nic_count == 16);

const scenario p6_b300_scenarios[] = {
	{ "all", p6_b300_nics, p6_b300_nic_count },
};

/* The static fixture registry. One entry per platform. */
const platform_fixture fixture_registry[] = {
	{ "p4d", "p4d/topology.xml", p4d_scenarios,
	  sizeof(p4d_scenarios) / sizeof(p4d_scenarios[0]) },
	{ "p5en", "p5en/topology.xml", p5en_scenarios,
	  sizeof(p5en_scenarios) / sizeof(p5en_scenarios[0]) },
	{ "p6-b200", "p6-b200/topology.xml", p6_b200_scenarios,
	  sizeof(p6_b200_scenarios) / sizeof(p6_b200_scenarios[0]) },
	{ "p6-b300", "p6-b300/topology.xml", p6_b300_scenarios,
	  sizeof(p6_b300_scenarios) / sizeof(p6_b300_scenarios[0]) },
	{ "p6e-gb200", "p6e-gb200/topology.xml", p6e_gb200_scenarios,
	  sizeof(p6e_gb200_scenarios) / sizeof(p6e_gb200_scenarios[0]) },
};

/*
 * Minimal libfabric NIC information used by the fixture runner.
 *
 * populate() calls fi_dupinfo(), which asks each fid_nic to duplicate itself
 * through fi_control(FI_DUP). Returning -FI_ENOSYS lets libfabric use its
 * shallow-copy fallback, preserving the PCI bus attributes needed to match the
 * NIC with its hwloc PCI device.
 */
struct test_nic_info {
	struct fi_info info;
	struct fid_nic nic;
	struct fi_bus_attr bus_attr;
	struct fi_ops nic_ops;
};

int test_nic_control(struct fid * /*fid*/, int /*command*/, void * /*arg*/)
{
	/* Let libfabric keep its shallow NIC copy and PCI bus attributes. */
	return -FI_ENOSYS;
}

/* True when two numeric PCI BDFs are identical. */
bool bdf_equal(const struct fi_pci_attr &a, uint16_t domain, uint8_t bus,
	       uint8_t device, uint8_t function)
{
	return a.domain_id == domain && a.bus_id == bus &&
	       a.device_id == device && a.function_id == function;
}

/*
 * Linear lookup of a NIC expectation by the leader's numeric BDF.
 *
 * The scenario arrays are small (<= 16 entries), so a typed linear scan beats
 * building and consulting a string-keyed map. Returns nullptr when no entry
 * matches the group leader's BDF.
 */
const nic_expectation *find_expectation(const scenario &sc,
					const struct fi_pci_attr &a)
{
	for (size_t i = 0; i < sc.num_nics; ++i) {
		const nic_expectation &e = sc.nics[i];
		if (bdf_equal(a, e.domain, e.bus, e.device, e.function)) {
			return &sc.nics[i];
		}
	}
	return nullptr;
}

/* Build an fi_info list from a scenario's NIC expectations. The returned list
 * and nested structures are owned by the backing vector; the topology class
 * only frees the fi_dupinfo() copies it makes. */
void build_nic_info_list(std::vector<test_nic_info> &nic_infos,
			 const nic_expectation *nics, size_t num_nics,
			 struct fi_info **out_list)
{
	nic_infos.clear();
	nic_infos.resize(num_nics);

	struct fi_info *head = nullptr;
	struct fi_info *tail = nullptr;

	for (size_t i = 0; i < num_nics; ++i) {
		test_nic_info &entry = nic_infos[i];
		memset(&entry.info, 0, sizeof(entry.info));
		memset(&entry.nic, 0, sizeof(entry.nic));
		memset(&entry.bus_attr, 0, sizeof(entry.bus_attr));
		memset(&entry.nic_ops, 0, sizeof(entry.nic_ops));

		entry.nic_ops.size = sizeof(entry.nic_ops);
		entry.nic_ops.control = test_nic_control;

		entry.nic.fid.fclass = FI_CLASS_NIC;
		entry.nic.fid.ops = &entry.nic_ops;

		entry.bus_attr.bus_type = FI_BUS_PCI;
		entry.bus_attr.attr.pci.domain_id = nics[i].domain;
		entry.bus_attr.attr.pci.bus_id = nics[i].bus;
		entry.bus_attr.attr.pci.device_id = nics[i].device;
		entry.bus_attr.attr.pci.function_id = nics[i].function;
		entry.nic.bus_attr = &entry.bus_attr;

		entry.info.nic = &entry.nic;
		entry.info.next = nullptr;
	}

	/* Link after the vector is fully populated so pointers are stable. */
	for (size_t i = 0; i < nic_infos.size(); ++i) {
		if (!head) {
			head = &nic_infos[i].info;
		} else {
			tail->next = &nic_infos[i].info;
		}
		tail = &nic_infos[i].info;
	}

	*out_list = head;
}

/* Resolve the fixture directory from the environment or the compile-time
 * define, so the test works in both in-tree and VPATH builds. */
std::string fixture_dir()
{
	const char *env = getenv("NCCL_OFI_TOPO_FIXTURE_DIR");
	if (env != nullptr && env[0] != '\0') {
		return std::string(env);
	}
#ifdef TOPO_FIXTURE_DIR
	return std::string(TOPO_FIXTURE_DIR);
#else
	return std::string(".");
#endif
}

/*
 * Select a topology source only while create() initializes and loads hwloc.
 * Restoring the prior value keeps the standalone test process composable with
 * other tests and with callers that already configured hwloc.
 */
class scoped_hwloc_xmlfile {
public:
	explicit scoped_hwloc_xmlfile(const std::string &path)
		: had_previous_(false), active_(false)
	{
		const char *previous = getenv("HWLOC_XMLFILE");
		if (previous != nullptr) {
			this->had_previous_ = true;
			this->previous_ = previous;
		}

		if (setenv("HWLOC_XMLFILE", path.c_str(), 1) != 0) {
			fprintf(stderr, "FAIL: cannot set HWLOC_XMLFILE\n");
			return;
		}
		this->active_ = true;
	}

	~scoped_hwloc_xmlfile()
	{
		if (!this->active_) {
			return;
		}

		int ret = this->had_previous_
			? setenv("HWLOC_XMLFILE", this->previous_.c_str(), 1)
			: unsetenv("HWLOC_XMLFILE");
		if (ret != 0) {
			fprintf(stderr, "FAIL: cannot restore HWLOC_XMLFILE\n");
		}
	}

	scoped_hwloc_xmlfile(const scoped_hwloc_xmlfile &) = delete;
	scoped_hwloc_xmlfile &operator=(const scoped_hwloc_xmlfile &) = delete;

	bool valid() const
	{
		return this->active_;
	}

private:
	std::string previous_;
	bool had_previous_;
	bool active_;
};

std::unique_ptr<nccl_ofi_topo_t> create_topology(const std::string &path)
{
	scoped_hwloc_xmlfile xmlfile(path);
	if (!xmlfile.valid()) {
		return nullptr;
	}
	return nccl_ofi_topo_t::create();
}

/* The leading NIC's PCI attributes of a group list, or nullptr if the leader
 * carries no PCI bus attributes. */
const struct fi_pci_attr *info_pci(struct fi_info *info)
{
	if (!info || !info->nic || !info->nic->bus_attr ||
	    info->nic->bus_attr->bus_type != FI_BUS_PCI) {
		return nullptr;
	}
	return &info->nic->bus_attr->attr.pci;
}

/* Count NIC members of a group list. */
int info_list_len(struct fi_info *info)
{
	int n = 0;
	for (struct fi_info *p = info; p != nullptr; p = p->next) {
		++n;
	}
	return n;
}

/* Format a numeric PCI BDF for diagnostics only. */
std::string bdf_text(const struct fi_pci_attr &a)
{
	char buf[32];
	snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%01x", a.domain_id, a.bus_id,
		 a.device_id, a.function_id);
	return std::string(buf);
}

/*
 * Run a single scenario: build the fi_info list, populate(), group(), iterate
 * group leaders, and verify queries and membership against the static arrays.
 */
bool run_scenario(const platform_fixture &fx, const std::string &xml_path,
		  const scenario &sc)
{
	/* fi_dupinfo() retains a shallow reference to each test NIC when
	 * FI_DUP is unsupported, so nic_infos must outlive the topology. */
	std::vector<test_nic_info> nic_infos;

	auto topo = create_topology(xml_path);
	if (!topo) {
		fprintf(stderr, "FAIL: [%s/%s] topology creation failed\n",
			fx.platform, sc.name);
		return false;
	}

	struct fi_info *info_list = nullptr;
	build_nic_info_list(nic_infos, sc.nics, sc.num_nics, &info_list);

	if (topo->populate(info_list) != 0) {
		fprintf(stderr, "FAIL: [%s/%s] populate failed\n",
			fx.platform, sc.name);
		return false;
	}
	if (topo->group() != 0) {
		fprintf(stderr, "FAIL: [%s/%s] group failed\n",
			fx.platform, sc.name);
		return false;
	}

	/* Walk group leaders through the public iterator. Track how many groups
	 * each supplied NIC appears in (indexed by position in the scenario
	 * array) so we can confirm every NIC lands in exactly one group. */
	std::vector<int> membership(sc.num_nics, 0);
	int num_leaders = 0;
	bool ok = true;

	nccl_ofi_topo_data_iterator_t iter;
	if (topo->set_to_begin(&iter) != 0) {
		fprintf(stderr, "FAIL: [%s/%s] set_to_begin failed\n",
			fx.platform, sc.name);
		return false;
	}

	struct fi_info *group = nullptr;
	while ((group = nccl_ofi_topo_next_info_list(&iter)) != nullptr) {
		++num_leaders;

		/* The leader is the first NIC of the group list. Grouping and
		 * rail sorting choose which NIC leads, so we read the leader
		 * from iteration rather than assuming a fixed one. */
		const struct fi_pci_attr *leader = info_pci(group);
		if (leader == nullptr) {
			fprintf(stderr,
				"FAIL: [%s/%s] group leader missing PCI attributes\n",
				fx.platform, sc.name);
			ok = false;
			continue;
		}

		/* Every member must be one of the supplied NICs. */
		for (struct fi_info *m = group; m != nullptr; m = m->next) {
			const struct fi_pci_attr *mp = info_pci(m);
			const nic_expectation *me =
				mp ? find_expectation(sc, *mp) : nullptr;
			if (me == nullptr) {
				fprintf(stderr,
					"FAIL: [%s/%s] unexpected NIC %s in a group\n",
					fx.platform, sc.name,
					mp ? bdf_text(*mp).c_str() : "<no-pci>");
				ok = false;
				continue;
			}
			membership[(size_t)(me - sc.nics)] += 1;
		}

		/* Query the public NIC-to-GPU interface for this leader and
		 * compare against the static expectation for its BDF. */
		const nic_expectation *exp = find_expectation(sc, *leader);
		if (exp == nullptr) {
			fprintf(stderr,
				"FAIL: [%s/%s] leader %s has no expectation\n",
				fx.platform, sc.name, bdf_text(*leader).c_str());
			ok = false;
		} else {
			bool result = false;
			if (topo->nic_gpu_share_pcie_switch(group, &result) != 0) {
				fprintf(stderr,
					"FAIL: [%s/%s] query failed for %s\n",
					fx.platform, sc.name,
					bdf_text(*leader).c_str());
				ok = false;
			} else if (result != exp->expected_share_pcie_switch) {
				fprintf(stderr,
					"FAIL: [%s/%s] leader %s share_pcie_switch=%s, expected %s\n",
					fx.platform, sc.name,
					bdf_text(*leader).c_str(),
					result ? "true" : "false",
					exp->expected_share_pcie_switch ? "true" : "false");
				ok = false;
			} else {
				printf("  [%s/%s] leader %-15s members=%d share_pcie_switch=%s\n",
				       fx.platform, sc.name,
				       bdf_text(*leader).c_str(), info_list_len(group),
				       result ? "true" : "false");
			}
		}
	}

	/* Every supplied NIC must appear in exactly one group. */
	for (size_t i = 0; i < sc.num_nics; ++i) {
		if (membership[i] != 1) {
			fprintf(stderr,
				"FAIL: [%s/%s] NIC %s (%02x:%02x.%x) appears in %d groups, expected 1\n",
				fx.platform, sc.name, sc.nics[i].nic,
				sc.nics[i].bus, sc.nics[i].device,
				sc.nics[i].function, membership[i]);
			ok = false;
		}
	}

	/* num_info_lists() must agree with the number of leaders iterated. */
	int reported = 0;
	if (topo->num_info_lists(&reported) != 0 || reported != num_leaders) {
		fprintf(stderr,
			"FAIL: [%s/%s] num_info_lists=%d but iterated %d leaders\n",
			fx.platform, sc.name, reported, num_leaders);
		ok = false;
	}

	if (ok) {
		printf("PASS: [%s/%s] %zu NICs -> %d group leaders, max_group_size=%d\n",
		       fx.platform, sc.name, sc.num_nics, num_leaders,
		       topo->max_group_size());
	}

	return ok;
}

/* Negative/invalid-argument coverage against the public query, exercised once
 * per fixture using a NIC that is absent from the topology. */
bool run_negative(const platform_fixture &fx, const std::string &xml_path)
{
	/* The populated NIC info is shallowly referenced by the topology's
	 * duplicated list, so keep its backing storage alive through destruction. */
	std::vector<test_nic_info> nic_infos;

	auto topo = create_topology(xml_path);
	if (!topo) {
		fprintf(stderr, "FAIL: [%s] topology creation failed\n",
			fx.platform);
		return false;
	}

	/* Populate/group with a real NIC so the topology is in a normal state. */
	const scenario &first = fx.scenarios[0];
	struct fi_info *info_list = nullptr;
	build_nic_info_list(nic_infos, first.nics, 1, &info_list);
	if (topo->populate(info_list) != 0 || topo->group() != 0) {
		fprintf(stderr, "FAIL: [%s] negative setup failed\n", fx.platform);
		return false;
	}

	bool ok = true;

	/* A NIC with a BDF that is not present in the topology must report
	 * false, not error and not a spurious PCIe-switch claim. */
	const nic_expectation absent = { "absent", 0x0000, 0xff, 0x00, 0x0, false };
	std::vector<test_nic_info> absent_nic_infos;
	struct fi_info *absent_list = nullptr;
	build_nic_info_list(absent_nic_infos, &absent, 1, &absent_list);

	bool result = true;
	if (topo->nic_gpu_share_pcie_switch(absent_list, &result) != 0) {
		fprintf(stderr, "FAIL: [%s] query errored on absent NIC\n",
			fx.platform);
		ok = false;
	} else if (result != false) {
		fprintf(stderr, "FAIL: [%s] absent NIC reported shared switch\n",
			fx.platform);
		ok = false;
	}

	/* Invalid arguments must be rejected without dereferencing them. */
	if (topo->nic_gpu_share_pcie_switch(nullptr, &result) == 0) {
		fprintf(stderr, "FAIL: [%s] null nic_info accepted\n", fx.platform);
		ok = false;
	}
	if (topo->nic_gpu_share_pcie_switch(absent_list, nullptr) == 0) {
		fprintf(stderr, "FAIL: [%s] null result accepted\n", fx.platform);
		ok = false;
	}

	if (ok) {
		printf("PASS: [%s] negative/invalid-argument queries\n", fx.platform);
	}
	return ok;
}

/* Run every scenario of one fixture, then its negative coverage, then a
 * repeated lifecycle pass to confirm create()/destructor ownership is clean. */
bool run_fixture(const platform_fixture &fx, const std::string &dir)
{
	std::string xml_path = dir + "/" + fx.xml_relpath;

	for (size_t i = 0; i < fx.num_scenarios; ++i) {
		if (!run_scenario(fx, xml_path, fx.scenarios[i])) {
			return false;
		}
	}

	if (!run_negative(fx, xml_path)) {
		return false;
	}

	/* Repeated construction/destruction of the same fixture must succeed,
	 * confirming create()/destructor ownership is clean. */
	for (int i = 0; i < 2; ++i) {
		if (!run_scenario(fx, xml_path, fx.scenarios[0])) {
			fprintf(stderr, "FAIL: [%s] repeat iteration %d failed\n",
				fx.platform, i);
			return false;
		}
	}

	return true;
}

} /* namespace */

int topo_fixture_run_all()
{
	std::string dir = fixture_dir();
	printf("Using fixture directory: %s\n", dir.c_str());

	const size_t num_fixtures =
		sizeof(fixture_registry) / sizeof(fixture_registry[0]);
	for (size_t i = 0; i < num_fixtures; ++i) {
		if (!run_fixture(fixture_registry[i], dir)) {
			return 1;
		}
	}

	return 0;
}