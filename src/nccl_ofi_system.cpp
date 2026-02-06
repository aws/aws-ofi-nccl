/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <cstring>
#include <arpa/inet.h>
#include <cstdint>
#include <stdexcept>

#include "nccl_ofi.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_system.h"

#ifndef SYSFS_PRODUCT_NAME_STR
#define SYSFS_PRODUCT_NAME_STR "/sys/devices/virtual/dmi/id/product_name"
#endif


static bool is_routable_interface(struct ifaddrs* ifa) {
	if (ifa->ifa_addr == nullptr) {
		return false;
	}

	/* Skip downed interfaces */
	if (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING)) {
		return false;
	}

	/* Skip loopbacl interfaces */
	if (ifa->ifa_flags & IFF_LOOPBACK) {
		return false;
	}

	/* Skip docker stuff and virtual bridges */
	if (strncmp(ifa->ifa_name, "docker", 6) == 0 ||
	    strncmp(ifa->ifa_name, "br-", 3) == 0 ||
	    strncmp(ifa->ifa_name, "veth", 4) == 0 ||
	    strncmp(ifa->ifa_name, "virbr", 5) == 0 ||
	    strncmp(ifa->ifa_name, "bridge", 6) == 0) {
		return false;
	}

	return true;
}

uint32_t nccl_ofi_get_unique_node_id(void)
{
	struct ifaddrs *ifaddr = nullptr;
	struct ifaddrs *ifa = nullptr;
	struct in6_addr ipv6_addr;
	uint32_t ip_addr = 0;
	bool found_ipv6 = false;

	if (getifaddrs(&ifaddr) == -1) {
		throw std::runtime_error("Failed to get interface addresses");
	}

	/* Look for non-loopback IPv4 addresses first */
	for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
		if (!is_routable_interface(ifa)) {
			continue;
		}

		if (ifa->ifa_addr->sa_family == AF_INET) {
			struct sockaddr_in *sin = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
			ip_addr = ntohl(sin->sin_addr.s_addr);
			freeifaddrs(ifaddr);
			return ip_addr;
		}
	}

	/* IPv4 no bueno. Find a non-loopback IPv6 interface */
	for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
		if (!is_routable_interface(ifa)) {
			continue;
		}

		if (ifa->ifa_addr->sa_family == AF_INET6) {
			struct sockaddr_in6 *sin6 = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr);
			ipv6_addr = sin6->sin6_addr;
			found_ipv6 = true;
			break;
		}
	}

	if (found_ipv6) {
		/* Beat it into a 32-bit field so the caller doesn't have to */
		uint32_t *addr_parts = (uint32_t *)ipv6_addr.s6_addr;
		ip_addr = ntohl(addr_parts[0] ^ addr_parts[1] ^ addr_parts[2] ^ addr_parts[3]);
	}

	freeifaddrs(ifaddr);

	if (!found_ipv6) {
		throw std::runtime_error("No suitable IPv4 or IPv6 interface found");
	}

	return ip_addr;
}

const char *nccl_net_ofi_get_product_name(void)
{
	char file[] = SYSFS_PRODUCT_NAME_STR;
	FILE *fd = NULL;
	char ch;
	size_t len = 0;
	size_t product_name_len = 64;
	static bool init = false;
	static char *product_name = NULL;
	static pthread_mutex_t product_name_mutex = PTHREAD_MUTEX_INITIALIZER;

	char* forced_pn = getenv("OFI_NCCL_FORCE_PRODUCT_NAME");
	if (forced_pn != NULL) {
		return forced_pn;
	}

	nccl_net_ofi_mutex_lock(&product_name_mutex);

	if (init) {
		nccl_net_ofi_mutex_unlock(&product_name_mutex);
		return product_name;
	}

	init = true;

	fd = fopen(file, "r");
	if (fd == NULL) {
		NCCL_OFI_WARN("Error opening file: %s", file);
		goto error;
	}

	product_name = (char *)malloc(sizeof(char) * product_name_len);
	if (product_name == NULL) {
		NCCL_OFI_WARN("Unable to allocate product name");
		goto error;
	}

	/* Read first line of the file, reallocing the buffer as necessary */
	while ((feof(fd) == 0) && (ferror(fd) == 0) && ((ch = fgetc(fd)) != '\n')) {
		product_name[len++] = ch;
		if (len >= product_name_len) {
			char *new_product_name = (char *)realloc(product_name, len + product_name_len);
			if (new_product_name == NULL) {
				NCCL_OFI_WARN("Unable to (re)allocate product name");
				goto error;
			}
			product_name = new_product_name;
		}
	}

	product_name[len] = '\0';

	if (ferror(fd)) {
		NCCL_OFI_WARN("Error reading file: %s", file);
		goto error;
	}

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Product Name is %s", product_name);

	goto exit;

error:
	if (product_name) {
		free(product_name);
		product_name = NULL;
	}

exit:
	if (fd) {
		fclose(fd);
	}

	nccl_net_ofi_mutex_unlock(&product_name_mutex);

	return product_name;
}
