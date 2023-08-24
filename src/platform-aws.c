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
#ifdef HAVE_RDMA_FI_EXT_H
#include <rdma/fi_ext.h>
#endif

#include "nccl_ofi.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_param.h"
#include "tracepoint.h"

static bool sendrecv_support_ll128 = false;
static bool write_support_ll128 = false;
static bool disable_native_rdma_check;
const char *platform_type;

struct ec2_platform_data {
	const char* name;
	const char* topology;
	int default_dup_conns;
	float latency;
} platform_data_map[] = {
	{
		.name = "p4d.24xlarge",
		.topology = "p4d-24xl-topo.xml",
		.default_dup_conns = 0,
		.latency = 75.0,
	},
	{
		.name = "p4de.24xlarge",
		.topology = "p4de-24xl-topo.xml",
		.default_dup_conns = 0,
		.latency = 75.0,
	},
	{
		.name = "p3dn.24xlarge",
		.topology = NULL,
		.default_dup_conns = 4,
		.latency = 150.0,
	},
	{
		.name = "p5.48xlarge",
		.topology = "p5.48xl-topo.xml",
		.default_dup_conns = 0,
		.latency = 75.0,
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

static ncclResult_t configure_nccl_proto(struct ec2_platform_data *platform_data)
{
	int ret = ncclSuccess;
	
	/* Explicitly set the simple protocol using the "NCCL_PROTO" environment
	 * variable whenever we know that the LL/LL128 protocols are not safe,
	 * such as on P4d/P4e.
	 *
	 * This only has impact on the Nvidia CUDA case, as the
	 * Tranium code does not use the LL/LL128 protocols.
	 */
	bool support_ll128_proto = sendrecv_support_ll128 || write_support_ll128;
	if (!support_ll128_proto) {
		if (!getenv("NCCL_PROTO")) {
			NCCL_OFI_INFO(NCCL_INIT, "Setting NCCL_PROTO to \"simple\"");
			int rc = setenv("NCCL_PROTO", "simple", 0);
			if (rc) {
				NCCL_OFI_WARN("Error setting NCCL_PROTO environment variable: %d", rc);
				ret = ncclSystemError;
				goto exit;
			}
		} else if (strcmp(getenv("NCCL_PROTO"), "simple")) {
			NCCL_OFI_WARN("NCCL_PROTO was set to \"LL/LL128\", but the Libfabric endpoint does not support 128 byte in-order aligned stores. This endpoint may corrupt data during communication");
		}
	}

exit:
	return ret;
}

static ncclResult_t validate_rdma_write(struct fid_ep *ep)
{
	int ret = ncclSuccess;
#if HAVE_DECL_FI_OPT_EFA_EMULATED_WRITE
	bool optval;
	size_t optlen = sizeof(optval);

	ret = fi_getopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_EMULATED_WRITE, &optval, &optlen);
	if(ret != 0 || optlen != sizeof(optval)) {
		NCCL_OFI_WARN("Couldn't get FI_OPT_EFA_EMULATED_WRITE. optlen: %lu, RC: %d, ERROR: %s",
			      optlen, ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto exit;
	}
	/* If the selected protocol is RDMA write and RDMA write is not
	 * supported for the endpoint, throw an error 
	 */
	else if (optval && 0 == strcmp("RDMA", nccl_ofi_selected_protocol)) {
		NCCL_OFI_WARN("FI_OPT_EFA_EMULATED_WRITE is true when the communication protocol is RDMA write.");
		ret = ncclSystemError;
		goto exit;
	}
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Get endpoint option FI_OPT_EFA_EMULATED_WRITE. optval: %d", 
		       optval);
#else
	NCCL_OFI_WARN("FI_OPT_EFA_EMULATED_WRITE not declared when the communication protocol is RDMA write.");
	ret = ncclSystemError;
	goto exit;
#endif
exit:
	return ret;
}

static ncclResult_t configure_sendrecv_inorder(struct fid_ep *ep, bool is_init)
{
	int ret = ncclSuccess;
#if HAVE_DECL_FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES
	bool optval = true;

	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT,
			FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES, 
			&optval, sizeof(optval));
	if (ret != 0 && ret != -FI_EOPNOTSUPP) {
		NCCL_OFI_WARN("Couldn't set FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto exit;
	}

	/* If this is called during plugin initialization, set the global flag
	 * sendrecv_support_ll128 to true if
	 * FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES could be set to true,
	 * otherwise keep it at its default value of false.
	 */
	if (is_init) {
		if (ret == 0) {
			sendrecv_support_ll128 = true;
		}
	}
	/* If an endpoint supported SENDRECV LL128 during plugin initialization
	 * but does not support it now, throw an error.
	 */
	else if (sendrecv_support_ll128 && ret == -FI_EOPNOTSUPP) {
		NCCL_OFI_WARN("SENDRECV LL128 not supported while it was supported during initialization.");
		ret = ncclSystemError;
		goto exit;
	}
	/* If an endpoint did not support SENDRECV LL128 during plugin
	 * initialization but supports it now, throw an error.
	 */
	else if (!sendrecv_support_ll128 && ret == 0) {
		NCCL_OFI_WARN("SENDRECV LL128 supported while it not supported during initialization.");
		ret = ncclSystemError;
		goto exit;
	}
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Set endpoint option FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES. optval: %d, RC: %d, ERROR: %s", 
		       optval, ret, fi_strerror(-ret));
	ret = ncclSuccess;
exit:
#endif
	return ret;
}

static ncclResult_t configure_write_inorder(struct fid_ep *ep, bool is_init)
{
	int ret = ncclSuccess;
#if HAVE_DECL_FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES
	bool optval = true;

	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT,
			FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES,
			&optval, sizeof(optval));
	if (ret != 0 && ret != -FI_EOPNOTSUPP) {
		NCCL_OFI_WARN("Couldn't set FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		ret = ncclSystemError;
		goto exit;
	}
	/* If this is called during plugin initialization, set the global flag
	 * write_support_ll128 to true if
	 * FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES could be set to true,
	 * otherwise keep it at its default value of false.
	 */
	if (is_init) {
		if (ret == 0) {
			write_support_ll128 = true;
		}
	}
	/* If an endpoint supported WRITE LL128 during plugin initialization but
	 * does not support it now, throw an error.
	 */
	else if (write_support_ll128 && ret == -FI_EOPNOTSUPP) {
		NCCL_OFI_WARN("WRITE LL128 not supported while it was supported during initialization.");
		ret = ncclSystemError;
		goto exit;
	}
	/* If an endpoint did not support SENDRECV LL128 during plugin
	 * initialization but supports it now, throw an error.
	 */
	else if (!write_support_ll128 && ret == 0) {
		NCCL_OFI_WARN("WRITE LL128 supported while it not supported during initialization.");
		ret = ncclSystemError;
		goto exit;
	}
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Set endpoint option FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES. optval: %d, RC: %d, ERROR: %s", 
		       optval, ret, fi_strerror(-ret));
	ret = ncclSuccess;
exit:
#endif
	return ret;
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

	platform_type = get_platform_type();
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

	/* Disable NVLS topology discovery.  There's a bug with EFA
	 * and NCCL 2.17/2.18 that is still under investigation that
	 * causes random failures due to memory corruption during
	 * initialization.  For now, skip that code.  We need to come
	 * back to this when the bug is fixed.
	 */
	if (getenv("NCCL_NVLS_ENABLE") == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Disabling NVLS support when using Libfabric on AWS.");
		rc = setenv("NCCL_NVLS_ENABLE", "0", 1);
		if (rc != 0) {
			NCCL_OFI_WARN("Unable to set NCCL_NVLS_ENABLE");
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

	disable_native_rdma_check = (bool) ofi_nccl_disable_native_rdma_check();

	if (ofi_nccl_net_latency() < 0) {
		if (platform_data && platform_data->latency >= 0.0) {
			net_latency = platform_data->latency;
		} else {
			/* For historical reasons, default value for EFA is 150 us */
			net_latency = 150.0;
		}
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Internode latency set at %.1f us",
				net_latency);
	}

exit:
	return ret;
}

ncclResult_t platform_config_endpoint(struct fi_info *info, struct fid_ep* endpoint) {
	static bool is_init = true;
	int ret = ncclSuccess;

	if (endpoint == NULL) {
		NCCL_OFI_WARN("Unable to configure invalid endpoint");
		ret = ncclSystemError;
		goto exit;
	}

	/* short circuit when not using EFA */
	if (0 != strcmp(info->fabric_attr->prov_name, "efa")) {
		ret = ncclSuccess;
		goto exit;
	}

	/* If the selected communication protocol is RDMA write and the user did
	 * not disable the native RDMA support check, validate that the
	 * FI_OPT_EFA_EMULATED_WRITE endpoint option can be accessed, and that
	 * emulated writes are disabled.
	 */
	if (0 == strcmp("RDMA", nccl_ofi_selected_protocol) && !disable_native_rdma_check) {
		ret = validate_rdma_write(endpoint);
		if (ret != 0) {
			goto exit;
		}
	}

#if HAVE_CUDA
	/* During initialization, if the chosen communication protocol is
	 * SENDRECV, try to set FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES
	 * to true to see if the LL/LL128 protocol is supported. After
	 * initialization, try to set the option to true again and if the
	 * LL/LL128 protocols are not supported for SENDRECV and were supported
	 * in initialization, throw an error.
	 */
	if (0 == strcmp("SENDRECV", nccl_ofi_selected_protocol)) {
		ret = configure_sendrecv_inorder(endpoint, is_init);
		if (ret != 0) {
			goto exit;
		}
	}

	/* During initialization, if the chosen communication protocol is RDMA, try to
	 * set FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES to true to see if the
	 * LL/LL128 protocol is supported. After initialization, try to set the option
	 * to true again and if the LL/LL128 protocols are not supported for RDMA and
	 * were supported in initialization, throw an error.
	 */
	else if (0 == strcmp("RDMA", nccl_ofi_selected_protocol)) {
		ret = configure_write_inorder(endpoint, is_init);
		if (ret != 0) {
			goto exit;
		}
	}

	/* if this is called during the plugin initialization, determine whether
	 * to explicitly set NCCL_PROTO to "simple" based on whether we support
	 * the LL/LL128 NCCL protocols.
	 */
	if (is_init) {
		struct ec2_platform_data *platform_data = get_platform_data(platform_type);
		ret = configure_nccl_proto(platform_data);
		if (ret != 0) {
			goto exit;
		}
	}
#endif // HAVE_CUDA
exit:
	/* if this is called during the plugin initialization, indicate that the
	 * intialzation has already completed and should not be done again 
	 */
	if (is_init) {
		is_init = false;
	}

	return ret;
}
