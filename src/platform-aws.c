/*
 * Copyright (c) 2018-2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#define _GNU_SOURCE
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "tracepoint.h"


struct ec2_platform_data {
	const char* name;
	const char* topology;
	int default_dup_conns;
	bool force_proto_simple;
} platform_data_map[] = {
	[0] = {
		.name = "p4d.24xlarge",
		.topology = "p4d-24xl-topo.xml",
		.default_dup_conns = 0,
		.force_proto_simple = true,
	},
	[1] = {
		.name = "p4de.24xlarge",
		.topology = "p4de-24xl-topo.xml",
		.default_dup_conns = 0,
		.force_proto_simple = true,
	},
	[2] = {
		.name = "p3dn.24xlarge",
		.topology = NULL,
		.default_dup_conns = 4,
		.force_proto_simple = true,
	},
};

/*
 * @brief	Provides EC2 platform type as reported by the
 * 		first line of
 *		/sys/devices/virtual/dmi/id/product_name.
 *		Users of this API *should* free the buffer when a
 *		Non-NULL string is returned.
 *
 * @return	NULL, on allocation and file system error
 * 		EC2 platform type, on success
 */
static const char* get_platform_type(void)
{
	char file[] = "/sys/devices/virtual/dmi/id/product_name";
	FILE *fd = NULL;
	char ch;
	size_t len = 0;
	size_t platform_type_len = 64;
	char *platform_type = NULL;

	fd = fopen(file, "r");
	if (fd == NULL) {
		NCCL_OFI_WARN("Error opening file: %s", file);
		goto error;
	}

	platform_type = (char *)malloc(sizeof(char)*platform_type_len);
	if (platform_type == NULL) {
		NCCL_OFI_WARN("Unable to allocate platform type");
		goto error;
	}

	/* Read first line of the file, reallocing the buffer as necessary */
	while ((feof(fd) == 0) && (ferror(fd) == 0) && ((ch = fgetc(fd)) != '\n')) {
		platform_type[len++] = ch;
		if (len >= platform_type_len) {
			platform_type = realloc(platform_type, len + platform_type_len);
		}
	}

	if (ferror(fd)) {
		NCCL_OFI_WARN("Error reading file: %s", file);
		goto error;
	}

	platform_type[len] = '\0';

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Read %d bytes. EC2 platform type is %s", len, platform_type);

	fclose(fd);
	return platform_type;

error:
	if (platform_type)
		free(platform_type);
	if (fd)
		fclose(fd);
	return platform_type;
}

/*
 * @brief	Returns static topology filename for given platform type, if found
 *
 * @input	Platform type
 *
 * @return	NULL, if no topology found
 * 		Topology filename, if match found
 */
struct ec2_platform_data *get_platform_data(const char *platform_type)
{
	const size_t platform_n = sizeof(platform_data_map)/sizeof(platform_data_map[0]);

	for (size_t idx = 0; idx < platform_n; idx++) {
		if (strcmp(platform_type, platform_data_map[idx].name) == 0)
			return &platform_data_map[idx];
	}

	return NULL;
}

/*
 * @brief	Update NCCL's system topology using static pre-configured topology
 * 		files for supported EC2 platform types.
 *
 * @return	0, when we are succesfully able to update NCCL topology or
 * 		   if we find no match
 * 		error, on failure
 */
ncclResult_t platform_init(void)
{
	int ret = ncclSuccess;
	int rc = 0;
	struct ec2_platform_data *platform_data;
	uint32_t libversion = 0;

	NCCL_OFI_INFO(NCCL_INIT, "Configuring AWS-specific options");

	const char *platform_type = get_platform_type();
	if (platform_type == NULL) {
		ret = ncclSystemError;
		goto exit;
	}

	platform_data = get_platform_data(platform_type);

	/* if we're here, we think we're on an EC2 instance, so force
	 * EFA provider (for platforms without EFA, this will cause a
	 * fallback to NCCL's internal TCP.  In the case of Neuron, a
	 * hard failure when there are no NICs.  Both are the
	 * behaviors we want).
	 */
	if (!getenv("FI_PROVIDER")) {
		NCCL_OFI_INFO(NCCL_INIT, "Setting provider_filter to efa");
		provider_filter = "efa";
	}

#if HAVE_CUDA
	/* Use the simple protocol whenever we're not sure the
	 * LL/LL128 protocols are safe.  In the future, we may want to
	 * revisit this and only set simple in cases where we know
	 * that it is not safe (P4d/P4e).
	 *
	 * This only has impact on the Nvidia CUDA case, as the
	 * Tranium code does not use the LL/LL128 protocols.
	 */
	if (!getenv("NCCL_PROTO") && (!platform_data || platform_data->force_proto_simple)) {
		NCCL_OFI_INFO(NCCL_INIT, "Setting NCCL_PROTO to \"simple\"");
		rc = setenv("NCCL_PROTO", "simple", 0);
		if (rc) {
			NCCL_OFI_WARN("Error setting NCCL_PROTO environment variable : %d", rc);
			ret = ncclSystemError;
			goto exit;
		}
	}
#endif

#if HAVE_CUDA
	/*
	 * FI_EFA_FORK_SAFE environment variable tells Libfabric to enable
	 * fork-safe support in legacy versions of the rdma-core library.
	 * Libfabric checks if additional handling is required for fork safety,
	 * and does not introduce this additional overhead of setting MADV_DONTFORK
	 * for new versions of rdma-core (38.0 and later) and the Linux kernel
	 * that support copy-on-fork for pinned memory (5.13 and later).
	 * These new versions are always fork-safe and additional support in userspace
	 * is not required.
	 *
	 * When legacy versions of the kernel and rdma-core are used, setting
	 * FI_EFA_FORK_SAFE to 1 disables the use of huge pages in Libfabric.
	 *
	 * To prevent data corruption, the EFA provider registers an atfork
	 * handler which will abort the process whenever it believes
	 * rdma-core is not fork-safe.
	 *
	 * NCCL applications heavily re-use the buffers for communication and
	 * thus are not sensitive to increased memory registration costs.
	 * To prevent NCCL based applications from getting aborted when using
	 * fork(), the plugin explicitly enables FI_EFA_FORK_SAFE environment
	 * variable, even in legacy environments where the overhead is high.
	 *
	 * The Neuron team has asked us to skip trying to set this
	 * environment variable on Neuron platforms, so we only do
	 * this for Nvidia platforms.
	 */
	libversion = fi_version();
	const char * fork_safe_var_name =
		(FI_MAJOR(libversion) > 1 || (FI_MAJOR(libversion) == 1 && FI_MINOR(libversion) >= 13))
		? "FI_EFA_FORK_SAFE"
		: "RDMAV_FORK_SAFE";
	if (!getenv(fork_safe_var_name)) {
		NCCL_OFI_INFO(NCCL_INIT, "Setting %s environment variable to 1", fork_safe_var_name);
		rc = setenv(fork_safe_var_name, "1", 1);
		if (rc != 0) {
			NCCL_OFI_WARN("Unable to set %s", fork_safe_var_name);
			ret = ncclSystemError;
			goto exit;
		}
	}
#endif

	/*
	 * Update topology if platform topology is available and 
	 * environment variable NCCL_TOPO_FILE is not set.
	 */
	if (getenv("NCCL_TOPO_FILE")) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Running on %s platform, NCCL_TOPO_FILE environment variable is already set to %s",
			      platform_type, getenv("NCCL_TOPO_FILE"));
	} else if (platform_data && platform_data->topology) {
		char topology_path[PATH_MAX];

		rc = snprintf(topology_path, sizeof(topology_path), "%s/%s",
				XML_DIR, platform_data->topology);
		if (rc < 0 || rc >= sizeof(topology_path)) {
			NCCL_OFI_WARN("Error occurred while forming the complete topology XML file path. RC: %d, Buffer Size: %d, XML dir: %s, Topology file: %s",
					rc, PATH_MAX, XML_DIR, platform_data->topology);
			ret = ncclSystemError;
			goto exit;
		}

		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"Running on %s platform, Setting NCCL_TOPO_FILE environment variable to %s",
				platform_type, topology_path);

		rc = setenv("NCCL_TOPO_FILE", topology_path, 1);
		if (rc != 0) {
			NCCL_OFI_WARN("Unable to set NCCL_TOPO_FILE");
			ret = ncclSystemError;
			goto exit;
		}

	}

	if (nic_dup_conns == 0 && platform_data)
		nic_dup_conns = platform_data->default_dup_conns;

exit:
	if (platform_type)
		free((char *)platform_type);
	return ret;
}
