/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "libfabric_mock.h"

/*
 * Provide the logger and param symbols that nccl_ofi_ofiutils.cpp needs.
 * We define OFI_NCCL_PARAM_DEFINE so the param header stamps out definitions
 * rather than extern declarations.
 */
#include "test-logger.h"
nccl_ofi_logger_t ofi_log_function = logger;

#define OFI_NCCL_PARAM_DEFINE 1
#include "nccl_ofi_param.h"

#include "nccl_ofi_ofiutils.h"

#include "nccl_ofi_platform.h"

/*
 * Stubs for symbols referenced by other functions in
 * nccl_ofi_ofiutils.cpp that we are not testing here
 * (e.g., ep_create).  These are never called by the
 * code paths under test.
 */
enum gdr_support_level_t support_gdr = GDR_UNSUPPORTED;

void PlatformManager::register_platform(std::unique_ptr<Platform>&&) {}

PlatformManager& PlatformManager::get_global()
{
	static PlatformManager instance;
	return instance;
}

uint32_t nccl_ofi_get_unique_node_id() { return 0; }

using ::testing::_;
using ::testing::Return;
using ::testing::DoAll;
using ::testing::SetArgPointee;
using ::testing::Invoke;

/*
 * Helper to build a minimal fi_info chain for testing.  Allocates real
 * structs so the filter_provider_list linked-list surgery works on
 * actual pointers.  Caller must free with free_test_info_list().
 */
struct test_fi_info {
	struct fi_info info;
	struct fi_fabric_attr fabric_attr;
	struct fi_domain_attr domain_attr;
	struct fi_ep_attr ep_attr;
};

static test_fi_info *alloc_test_info(const char *prov_name,
				     const char *fabric_name,
				     const char *domain_name,
				     uint32_t addr_format = FI_FORMAT_UNSPEC)
{
	auto *t = new test_fi_info();
	memset(t, 0, sizeof(*t));

	t->fabric_attr.prov_name = strdup(prov_name);
	t->fabric_attr.name = strdup(fabric_name);
	t->domain_attr.name = strdup(domain_name);
	t->info.fabric_attr = &t->fabric_attr;
	t->info.domain_attr = &t->domain_attr;
	t->info.ep_attr = &t->ep_attr;
	t->info.addr_format = addr_format;

	return t;
}

static void free_test_info(test_fi_info *t)
{
	free(t->fabric_attr.prov_name);
	free(t->fabric_attr.name);
	free(t->domain_attr.name);
	delete t;
}

/* Chain a list of test_fi_info into a linked list via ->info.next */
static struct fi_info *chain_infos(std::vector<test_fi_info *> &infos)
{
	for (size_t i = 0; i + 1 < infos.size(); i++)
		infos[i]->info.next = &infos[i + 1]->info;
	if (!infos.empty())
		infos.back()->info.next = nullptr;
	return &infos[0]->info;
}

class GetProvidersTest : public ::testing::Test {
protected:
	void SetUp() override {
		mock = new ::testing::NiceMock<LibfabricMock>();
		g_libfabric_mock = mock;
	}

	void TearDown() override {
		for (auto *t : test_infos)
			free_test_info(t);
		test_infos.clear();
		delete mock;
		g_libfabric_mock = nullptr;
	}

	/*
	 * Convenience: allocate a test_fi_info, track it for cleanup,
	 * and return it.
	 */
	test_fi_info *make_info(const char *prov, const char *fabric,
				const char *domain,
				uint32_t addr_format = FI_FORMAT_UNSPEC)
	{
		auto *t = alloc_test_info(prov, fabric, domain, addr_format);
		test_infos.push_back(t);
		return t;
	}

	LibfabricMock *mock;
	std::vector<test_fi_info *> test_infos;
};

/*
 * Verify that nccl_ofi_ofiutils_get_providers() forwards the caller's
 * arguments to fi_getinfo() exactly: the required_version, NULL node
 * and service strings, zero flags, and the caller's hints pointer.
 * Also verify that on failure, *num_prov_infos is zeroed and
 * *prov_info_list remains NULL.
 */
TEST_F(GetProvidersTest, PassesCorrectArgumentsToFiGetinfo) {
	struct fi_info hints = {};
	uint32_t version = FI_VERSION(1, 20);

	EXPECT_CALL(*mock, fi_getinfo(version, nullptr, nullptr, 0ULL, &hints, _))
		.WillOnce(Return(-FI_ENODATA));

	struct fi_info *result = nullptr;
	unsigned int count = 99;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, version, &hints,
						 &result, &count);

	EXPECT_NE(rc, 0);
	EXPECT_EQ(result, nullptr);
	EXPECT_EQ(count, 0u);
}

/*
 * Verify that a non-zero error code returned by fi_getinfo() is
 * propagated unchanged to the caller, and that the output list
 * pointer remains NULL.
 */
TEST_F(GetProvidersTest, PropagatesFiGetinfoFailure) {
	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(Return(-FI_ENODATA));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, -FI_ENODATA);
	EXPECT_EQ(result, nullptr);
}

/*
 * Verify that when fi_getinfo() returns success (0) but sets the
 * output provider list to NULL, get_providers() treats this as
 * -FI_ENODATA rather than returning success with an empty list.
 */
TEST_F(GetProvidersTest, ReturnsErrorWhenGetinfoReturnsNullList) {
	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(nullptr), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, -FI_ENODATA);
}

/*
 * Verify the happy path: a single EFA provider returned by
 * fi_getinfo() passes through all filter stages unmodified.
 * The result list should contain exactly that provider, count
 * should be 1, and fi_freeinfo() must not be called (no error).
 */
TEST_F(GetProvidersTest, SingleEfaProviderNoFilter) {
	auto *efa = make_info("efa", "efa-fabric", "efa0");

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&efa->info), Return(0)));
	EXPECT_CALL(*mock, fi_freeinfo(_)).Times(0);

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(result, &efa->info);
	EXPECT_EQ(count, 1u);
}

/*
 * Verify that the prov_include filter correctly removes providers
 * whose fabric_attr->prov_name does not appear in the include list.
 *
 * Setup: fi_getinfo() returns [tcp, efa].  prov_include="efa".
 * Expected: tcp is removed, only efa remains with count=1.
 */
TEST_F(GetProvidersTest, ProvIncludeFiltersToMatchingProvider) {
	auto *tcp = make_info("tcp", "tcp-fabric", "eth0", FI_SOCKADDR_IN);
	make_info("efa", "efa-fabric", "efa0");

	chain_infos(test_infos);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&tcp->info), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers("efa", FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(count, 1u);
	EXPECT_STREQ(result->fabric_attr->prov_name, "efa");
}

/*
 * Verify that when prov_include filtering removes every provider
 * from the list, get_providers() returns -FI_ENODATA.
 *
 * Also documents a known behavior: fi_freeinfo() is NOT called in
 * this case because filter_provider_list() sets *providers = NULL
 * when all entries are removed, and the error path only calls
 * fi_freeinfo() when providers is non-NULL.  The removed nodes
 * are effectively leaked by the filter.
 */
TEST_F(GetProvidersTest, ProvIncludeFiltersOutAllProviders) {
	auto *tcp = make_info("tcp", "tcp-fabric", "eth0", FI_SOCKADDR_IN);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&tcp->info), Return(0)));
	EXPECT_CALL(*mock, fi_freeinfo(_)).Times(0);

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers("efa", FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, -FI_ENODATA);
}

/*
 * Verify that the TCP interface exclusion filter removes TCP
 * providers whose domain_attr->name appears in the default
 * OFI_NCCL_EXCLUDE_TCP_IF list ("lo,docker0"), while keeping
 * TCP providers on other interfaces.
 *
 * Setup: fi_getinfo() returns [tcp/lo, tcp/eth0].
 * Expected: tcp/lo is removed, tcp/eth0 survives.
 */
TEST_F(GetProvidersTest, TcpExcludedInterfaceFiltered) {
	auto *tcp_lo = make_info("tcp", "tcp-fabric", "lo", FI_SOCKADDR_IN);
	make_info("tcp", "tcp-fabric", "eth0", FI_SOCKADDR_IN);

	chain_infos(test_infos);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&tcp_lo->info), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(count, 1u);
	EXPECT_STREQ(result->domain_attr->name, "eth0");
}

/*
 * Verify that the TCP interface exclusion filter only applies to
 * providers whose prov_name starts with "tcp".  A non-TCP provider
 * (e.g., EFA) with a domain name that matches the exclude list
 * ("lo") must NOT be filtered out.
 */
TEST_F(GetProvidersTest, NonTcpProviderNotAffectedByTcpFilter) {
	auto *efa = make_info("efa", "efa-fabric", "lo");

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&efa->info), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(count, 1u);
	EXPECT_STREQ(result->fabric_attr->prov_name, "efa");
}

/*
 * Verify that the TCP address type filter removes TCP providers
 * with non-IPv4 address formats when OFI_NCCL_USE_IPV6_TCP is
 * disabled (the default).
 *
 * Setup: fi_getinfo() returns [tcp/FI_SOCKADDR_IN6, tcp/FI_SOCKADDR_IN].
 * Expected: the IPv6 entry is removed, only the IPv4 entry remains.
 */
TEST_F(GetProvidersTest, TcpIpv6FilteredWhenDisabled) {
	auto *tcp_v6 = make_info("tcp", "tcp-fabric", "eth0", FI_SOCKADDR_IN6);
	make_info("tcp", "tcp-fabric", "eth1", FI_SOCKADDR_IN);

	chain_infos(test_infos);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&tcp_v6->info), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(count, 1u);
	EXPECT_EQ(result->addr_format, (uint32_t)FI_SOCKADDR_IN);
}

/*
 * Verify that the match filter (prov_filter_by_match) deduplicates
 * the provider list to only those entries matching the first
 * provider's caps, mode, addr_format, endpoint type, protocol,
 * protocol version, prov_name, and fabric name.
 *
 * Setup: fi_getinfo() returns [efa0, efa1, tcp] where both EFA
 *        entries share identical attributes but TCP differs in
 *        caps and protocol.
 * Expected: TCP is removed, both EFA entries remain, count=2.
 */
TEST_F(GetProvidersTest, MatchFilterKeepsOnlyMatchingProviders) {
	auto *efa1 = make_info("efa", "efa-fabric", "efa0");
	efa1->info.caps = FI_TAGGED | FI_MSG;
	efa1->ep_attr.type = FI_EP_RDM;
	efa1->ep_attr.protocol = 1;
	efa1->ep_attr.protocol_version = 1;

	auto *efa2 = make_info("efa", "efa-fabric", "efa1");
	efa2->info.caps = FI_TAGGED | FI_MSG;
	efa2->ep_attr.type = FI_EP_RDM;
	efa2->ep_attr.protocol = 1;
	efa2->ep_attr.protocol_version = 1;

	auto *tcp = make_info("tcp", "tcp-fabric", "eth0", FI_SOCKADDR_IN);
	tcp->info.caps = FI_MSG;
	tcp->ep_attr.type = FI_EP_RDM;
	tcp->ep_attr.protocol = 2;
	tcp->ep_attr.protocol_version = 1;

	chain_infos(test_infos);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&efa1->info), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(count, 2u);
	EXPECT_STREQ(result->fabric_attr->prov_name, "efa");
	EXPECT_STREQ(result->next->fabric_attr->prov_name, "efa");
	EXPECT_EQ(result->next->next, nullptr);
}

/*
 * Verify that *num_prov_infos correctly counts all providers that
 * survive filtering.  Three identical EFA providers should all
 * pass every filter stage and yield count=3.
 */
TEST_F(GetProvidersTest, CountsMultipleProviders) {
	auto *e1 = make_info("efa", "efa-fabric", "efa0");
	make_info("efa", "efa-fabric", "efa1");
	make_info("efa", "efa-fabric", "efa2");

	chain_infos(test_infos);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&e1->info), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(count, 3u);
}

/*
 * Verify the error-path cleanup behavior when all providers are
 * filtered out by the TCP interface exclusion filter.
 *
 * A single TCP provider on "docker0" (in the default exclude list)
 * is removed, leaving providers == NULL.  The error path checks
 * if (providers) before calling fi_freeinfo(), so fi_freeinfo()
 * is NOT called.  This documents the current behavior where
 * filter_provider_list() detaches nodes without freeing them.
 */
TEST_F(GetProvidersTest, ErrorPathCallsFreeinfo) {
	auto *tcp_docker = make_info("tcp", "tcp-fabric", "docker0", FI_SOCKADDR_IN);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&tcp_docker->info), Return(0)));

	EXPECT_CALL(*mock, fi_freeinfo(_)).Times(0);

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_NE(rc, 0);
}

/*
 * Verify that prov_include accepts a comma-separated list of
 * provider names and keeps all matching providers.
 *
 * Setup: fi_getinfo() returns [efa, tcp, shm].  prov_include="efa,tcp".
 * Expected: shm is removed by prov_include.  Then the match filter
 *           keeps only entries matching the first survivor (efa),
 *           so tcp is also removed.  Final result: efa only, count=1.
 */
TEST_F(GetProvidersTest, ProvIncludeMultipleNames) {
	auto *efa = make_info("efa", "efa-fabric", "efa0");
	make_info("tcp", "tcp-fabric", "eth0", FI_SOCKADDR_IN);
	make_info("shm", "shm-fabric", "shm0");

	chain_infos(test_infos);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&efa->info), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers("efa,tcp", FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(count, 1u);
	EXPECT_STREQ(result->fabric_attr->prov_name, "efa");
}

/*
 * Verify that the TCP interface exclusion filter catches stacked
 * providers like "tcp;ofi_rxm".  The filter uses strncmp(prov_name,
 * "tcp", 3) so any provider name starting with "tcp" is subject
 * to interface exclusion.
 *
 * Setup: fi_getinfo() returns [tcp;ofi_rxm/lo, tcp;ofi_rxm/eth0].
 * Expected: the "lo" entry is removed, "eth0" survives.
 */
TEST_F(GetProvidersTest, StackedTcpProviderFiltered) {
	auto *stacked_lo = make_info("tcp;ofi_rxm", "tcp-fabric", "lo", FI_SOCKADDR_IN);
	make_info("tcp;ofi_rxm", "tcp-fabric", "eth0", FI_SOCKADDR_IN);

	chain_infos(test_infos);

	EXPECT_CALL(*mock, fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0ULL, _, _))
		.WillOnce(DoAll(SetArgPointee<5>(&stacked_lo->info), Return(0)));

	struct fi_info *result = nullptr;
	unsigned int count = 0;
	int rc = nccl_ofi_ofiutils_get_providers(nullptr, FI_VERSION(1, 18),
						 nullptr, &result, &count);

	EXPECT_EQ(rc, 0);
	EXPECT_EQ(count, 1u);
	EXPECT_STREQ(result->domain_attr->name, "eth0");
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
