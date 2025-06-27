/*

 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <algorithm>
#include <alloca.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#include <fstream>
#include <cstdint>
#include <system_error>
#include <map>
#include <mutex>
#include <string>

#ifdef HAVE_RDMA_FI_EXT_H
#include <rdma/fi_ext.h>
#endif
#include <regex.h>
#include <dlfcn.h>

#include "nccl_ofi.h"
#include "nccl_ofi_platform.h"
#include "platform-aws.h"
#include "nccl_ofi_environ.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_pthread.h"
#include "nccl_ofi_system.h"


/*
 * platform_data_map is an ordered list of platform entries.  The
 * name is evaluated as a POSIX regex, so entries like "p5.*" is
 * valid.  First match found wins, even if there is a more specific
 * match later in the array.
 *
 * Notes on the environment:
 *
 *  * Certain GPU architectures do not require a network flush, but
 *    NCCL versions <2.19.1 still enable flush by default on any
 *    GPU type.  For GPU generations earlier than Hopper, NCCL
 *    always enables flush, while for Hopper GPUs flush is enabled
 *    or disabled depending on the value of the
 *    NCCL_NET_FORCE_FLUSH environment variable. The default value
 *    for this variable is 1 for NCCL versions <2.19.1, which
 *    forces flush when it is not needed, so it is safe to set it
 *    to 0 if it is not explicitly set.
 *
 *  * NCCL v2.19.3 reduced the chunk size used when running NVLS Tree
 *    algorithm on greater than 4 nodes to 64KiB. This drastically impacted
 *    performance on AWS (Ref: https://github.com/NVIDIA/nccl/pull/1112/
 *    for some data). NCCL v2.20.3 has made this a tunable. Based on
 *    empirical testing, a max chunk size of 512KiB recovers from the
 *    regression and was also observed to be the default in v2.19.3.
 *    Setting this unconditionally without relying on ncclGetVersion symbol
 *    being available, since the parameter did not exist in versions prior
 *    to v2.20.
 *
 *    The NVLSTree chunk size can not be larger than the NVLS chunk size,
 *    so we ensure both are set to 512KiB.
 */
static struct ec2_platform_data platform_data_map[] = {
	{
		.name = "p4d.24xlarge",
		.regex = NULL,
		.topology = "p4d-24xl-topo.xml",
		.default_dup_conns = 0,
		.latency = 75.0,
		.gdr_required = true,
		.default_protocol = "SENDRECV",
		.domain_per_thread = 0,
		.env = {
			{ "NCCL_BUFFSIZE", "8388608" },
			{ "NCCL_P2P_NET_CHUNKSIZE", "524288" },
		},
	},
	{
		.name = "p4de.24xlarge",
		.regex = NULL,
		.topology = "p4de-24xl-topo.xml",
		.default_dup_conns = 0,
		.latency = 75.0,
		.gdr_required = true,
		.default_protocol = "SENDRECV",
		.domain_per_thread = 0,
		.env = {
			{ "NCCL_BUFFSIZE", "8388608" },
			{ "NCCL_P2P_NET_CHUNKSIZE", "524288" },
		},
	},
	{
		.name = "p3dn.24xlarge",
		.regex = NULL,
		.topology = NULL,
		.default_dup_conns = 4,
		.latency = 150.0,
		.gdr_required = false,
		.default_protocol = "SENDRECV",
		.domain_per_thread = 0,
		.env = {},
	},
	{
		.name = "p-series",
		/*
		 * we only want to match P5en and later, as earlier
		 * platforms all either need to be ignored or special
		 * cased.
		 */
		.regex = "^(p5en\\.48xlarge)|(^p([6-9]|[0-9]{2,}).*)",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 35.0,
		.gdr_required = true,
		.default_protocol = "RDMA",
		.domain_per_thread = 0,
		.env = {
			{ "NCCL_BUFFSIZE", "8388608" },
			{ "NCCL_P2P_NET_CHUNKSIZE", "524288" },
			{ "NCCL_NVLSTREE_MAX_CHUNKSIZE", "524288" },
			{ "NCCL_NVLS_CHUNKSIZE", "524288" },
			{ "NCCL_NET_FORCE_FLUSH", "0" },
		},
	},
	{
		.name = "p5/p5e",
		.regex = "^p5(e?\\..*)",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 75.0,
		.gdr_required = true,
		.default_protocol = "RDMA",
		.domain_per_thread = 0,
		.env = {
			{ "NCCL_BUFFSIZE", "8388608" },
			{ "NCCL_P2P_NET_CHUNKSIZE", "524288" },
			{ "NCCL_NVLSTREE_MAX_CHUNKSIZE", "524288" },
			{ "NCCL_NVLS_CHUNKSIZE", "524288" },
			{ "NCCL_NET_FORCE_FLUSH", "0" },
		},
	},
	{
		.name = "g5.48xlarge",
		.regex = NULL,
		.topology = "g5.48xl-topo.xml",
		.default_dup_conns = 0,
		.latency = 75.0,
		.gdr_required = false,
		.default_protocol = "SENDRECV",
		.domain_per_thread = 0,
		.env = {},
	},
	{
		.name = "trn1",
		.regex = "^trn1.*",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 75.0,
		.gdr_required = true,
		.default_protocol = "SENDRECV",
		.domain_per_thread = 1,
		.env = {},
	},
	{
		.name = "trn2",
		.regex = "^trn2.*",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 75.0,
		.gdr_required = true,
		.default_protocol = "RDMA",
		.domain_per_thread = 1,
		.env = {},
	},
	{
		.name = "inf",
		.regex = "^inf.*",
		.topology = NULL,
		.default_dup_conns = 0,
		.latency = 75.0,
		.gdr_required = true,
		.default_protocol = "SENDRECV",
		.domain_per_thread = 1,
		.env = {},
	},
};


/*
 * We need to cache the fields that we grabbed for each device so we don't go
 * read sysfs for each field that we need.
 */
static std::unordered_map<std::string, struct platform_aws_node_guid> guid_cache;
static std::mutex cache_mutex;

struct ec2_platform_data *platform_aws_get_platform_map(size_t *len)
{
	*len = sizeof(platform_data_map)/sizeof(platform_data_map[0]);
	return platform_data_map;
}


/*
 * internal function (exported for unit test purposes) for finding the
 * correct platform data entry.  You should use
 * platform_Aws_get_platform_data() so that you get caching and all
 * that niceness.
 */
struct ec2_platform_data *platform_aws_get_platform_entry(const char *platform_type,
							  struct ec2_platform_data *platform_data_list,
							  size_t platform_data_len)
{
	struct ec2_platform_data *response = NULL;
	regex_t regex;
	int ret;

	for (size_t idx = 0; idx < platform_data_len; idx++) {
		if (platform_data_list[idx].regex == NULL) {
			if (0 == strcmp(platform_type,
					platform_data_list[idx].name)) {
				response = &platform_data_list[idx];
				break;
			}
		} else {
			ret = regcomp(&regex, platform_data_list[idx].regex, REG_EXTENDED);
			if (ret != 0) {
				NCCL_OFI_WARN("Could not compile platform_type regex for %s",
					      platform_data_list[idx].regex);
				goto done;
			}

			ret = regexec(&regex, platform_type, 0, NULL, 0);

			regfree(&regex);

			if (ret == 0) {
				response = &platform_data_list[idx];
				break;
			} else if (ret != REG_NOMATCH) {
				NCCL_OFI_WARN("Regex match failed");
				goto done;
			}
		}
	}

done:
	NCCL_OFI_TRACE(NCCL_NET | NCCL_INIT, "Using platform block %s for instance type %s",
		      (response == NULL) ? "none" : response->name, platform_type);

	return response;
}


/*
 * @brief	Returns platform data for current platform type, if found
 *
 * @input	none
 *
 * @return	NULL, if no entry found
 * 		platform data, if match found
 */
static struct ec2_platform_data *get_platform_data(void)
{
	static bool init = false;
	static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
	static struct ec2_platform_data *platform_data = NULL;
	const char* platform_type = NULL;
	struct ec2_platform_data *platform_data_list;
	size_t platform_data_len;

	nccl_net_ofi_mutex_lock(&mutex);

	if (init) {
		goto done;
	}
	init = true;

	platform_type = nccl_net_ofi_get_product_name();
	if (platform_type == NULL) {
		goto done;
	}

	platform_data_list = platform_aws_get_platform_map(&platform_data_len);
	if (platform_data_list == NULL) {
		goto done;
	}

	platform_data = platform_aws_get_platform_entry(platform_type, platform_data_list,
							platform_data_len);

done:
	nccl_net_ofi_mutex_unlock(&mutex);

	return platform_data;
}


/*
 * validate that EFA is using RDMA write natively and not in an
 * emulated fasion.
 */
static int validate_rdma_write(struct fid_ep *ep)
{
	int ret = 0;
#if HAVE_DECL_FI_OPT_EFA_EMULATED_WRITE
	bool optval;
	size_t optlen = sizeof(optval);

	ret = fi_getopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_EFA_EMULATED_WRITE, &optval, &optlen);
	if (ret != 0) {
		NCCL_OFI_WARN("Couldn't get FI_OPT_EFA_EMULATED_WRITE. RC: %d, ERROR: %s",
			      ret, fi_strerror(-ret));
		goto exit;
	} else if (optlen != sizeof(optval)) {
		NCCL_OFI_WARN("Unexpected response size when checking FI_OPT_EFA_EMULATED_WRITE.  Expected %lu, got %lu",
			      sizeof(optval), optlen);
		ret = -EINVAL;
		goto exit;
	} else if (optval) {
		NCCL_OFI_WARN("FI_OPT_EFA_EMULATED_WRITE is true when the communication protocol is RDMA write.");
		ret = -EINVAL;
		goto exit;
	}
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Get endpoint option FI_OPT_EFA_EMULATED_WRITE. optval: %d",
		       optval);
#else
	NCCL_OFI_WARN("FI_OPT_EFA_EMULATED_WRITE not declared when the communication protocol is RDMA write.");
	ret = -EINVAL;
	goto exit;
#endif

exit:
	return ret;
}


#if HAVE_CUDA
/*
 * Try to set one of the in-order flags for either send/recv or rdma
 * on the current endpoint to true.  have_ordering will be the
 * returned value on output.
 *
 * Returns 0 on success (ie, have_ordering is in a sane state) or
 * -error code on unexpected failure.
 */
static int configure_ep_inorder(struct fid_ep *ep, int optname, const char* optname_name,
				bool *have_ordering)
{
#if HAVE_DECL_FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES
	int ret = 0;
	bool optval = true;

	*have_ordering = false;

	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT,
			optname, &optval, sizeof(optval));
	if (ret == -FI_EOPNOTSUPP || ret == -FI_ENOPROTOOPT) {
		NCCL_OFI_INFO(NCCL_INIT, "Setting %s not supported.", optname_name);
	} else if (ret != 0) {
		NCCL_OFI_WARN("Could not set %s. RC: %d, ERROR: %s",
			      optname_name, ret, fi_strerror(-ret));
		return ret;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT, "Setting %s have_ordering is true.", optname_name);
		*have_ordering = true;
	}

	NCCL_OFI_TRACE(NCCL_INIT, "fi_setopt(%s) ordering result %s, error code %d",
		       optname_name, have_ordering ? "yes" : "no", ret);
#else
	*have_ordering = false;
#endif
	return 0;
}

/*
 * For the RDMA protocol, try to set max msg size on the current endpoint
 * to the size of the max message we send with fi_send. This allows the EFA
 * provider to enable the zero-copy path.
 *
 * Returns 0 on success or -error code on unexpected failure.
 */
static int configure_ep_max_msg_size(struct fid_ep *ep)
{
	int ret = 0;

#if HAVE_DECL_FI_OPT_MAX_MSG_SIZE
	ssize_t eager_max_size = (ssize_t)ofi_nccl_eager_max_size();
	size_t optval = sizeof(nccl_ofi_rdma_connection_info_t);

	if (eager_max_size > 0) {
		optval = std::max(optval, static_cast<size_t>(eager_max_size));
	}

	ret = fi_setopt(&ep->fid, FI_OPT_ENDPOINT, FI_OPT_MAX_MSG_SIZE, &optval, sizeof(optval));

	NCCL_OFI_TRACE(NCCL_INIT, "fi_setopt(FI_OPT_MAX_MSG_SIZE) RC: %d", ret);

	if (ret == -FI_EOPNOTSUPP || ret == -FI_ENOPROTOOPT) {
		NCCL_OFI_INFO(NCCL_INIT, "Setting FI_OPT_MAX_MSG_SIZE not supported.");
		ret = 0;
	} else if (ret != 0) {
		NCCL_OFI_WARN("Could not set FI_OPT_MAX_MSG_SIZE. RC: %d, ERROR: %s", ret, fi_strerror(-ret));
	}
#endif

	return ret;
}


typedef ncclResult_t (*nccl_get_version_fn_t)(int *version);

static int configure_nvls_option(void)
{
	/* Disable NVLS topology discovery for older NCCL versions. There's a
	 * bug with EFA and NCCL version 2.18.3 and earlier on platforms with
	 * NVLink Switch support.  We selectively disable NVLS support
	 * to avoid the bug, which was fixed in 2.18.5.
	 */
	nccl_get_version_fn_t nccl_get_version = NULL;
	int version = 0;
	ncclResult_t nccl_ret;

	if (getenv("NCCL_NVLS_ENABLE") == NULL) {
		nccl_get_version = (nccl_get_version_fn_t)dlsym(RTLD_DEFAULT, "ncclGetVersion");
		if (nccl_get_version == NULL) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			    "Could not find ncclGetVersion symbol; skipping NVLS NCCL version check");
			return 0;
		} else {
			nccl_ret = nccl_get_version(&version);
			if (nccl_ret != ncclSuccess) {
				NCCL_OFI_WARN("ncclGetVersion returned %d", nccl_ret);
				return -ENOTSUP;
			}

			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "ncclGetVersion results = %d", version);
		}

		/* 2.18.5 */
		if (version < 21805) {
			NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Disabling NVLS support due to NCCL version %d", version);
			env_manager::getInstance().insert_envvar("NCCL_NVLS_ENABLE", "0", false);
		} else {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Not disabling NVLS support due to NCCL version %d", version);
		}
	}

	return 0;
}

static int configure_tuner()
{
	// NCCL has a bug in their handling of the combined net/tuner dso (at
	// least up to NCCL 2.27) where the tuner init does not open the net
	// shared library but later tries to close it, meaning that the shared
	// library gets closed and potentially unloaded from memory before NCCL
	// is done using the net part of the interface, which results in obvious
	// badness.
	//
	// If NCCL_TUNER_PLUGIN is set, NCCL will dlopen() the library whether
	// or not it is the same underlying library as the NET plugin, meaning
	// that we get the reference counting behavior we need.  So attempt to
	// always set NCCL_TUNER_PLUGIN to prevent the dlopen refcount bug.
	//
	// When this bug is fixed, we should use the next api version bump as a
	// way of shutting off this code.
	if (getenv("NCCL_TUNER_PLUGIN") != NULL) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "NCCL_TUNER_PLUGIN set; skipping configuration");
		return 0;
	}

	// if we know how the net plugin was found, use that value for the tuner
	// plugin.  Skip the case where a list is provide, because parsing the
	// list does not sound like a winning strategy and the only version of
	// NCCL that currently supports a comma delineated list also doesn't
	// support specifying full paths (meaning that the dladdr code below
	// will work fine).
	char *env_net_value = getenv("NCCL_NET_PLUGIN");
	if (env_net_value != NULL && strchr(env_net_value, ',') == NULL) {
		env_manager::getInstance().insert_envvar("NCCL_TUNER_PLUGIN", env_net_value, false);
		return 0;
	}

	// NCCL found the net plugin without setting NCCL_NET_PLUGIN.  Need to
	// figure out where we came from.
	Dl_info info;
	int rc = dladdr((void *)configure_tuner, &info);
	if (rc != 0) {
		// unlike every other call, rc == 0 is the error condition for
		// dladdr().
		//
		// NCCL 2.27 NCCL_TUNER_PLUGIN must be just the basename, without a path.
		const std::string net_pathname(info.dli_fname);
		auto const pos = net_pathname.find_last_of('/');
		if (pos == std::string::npos) {
			NCCL_OFI_WARN("Failed to parse shared library info.  Not configuring NCCL_TUNER_PLUGIN.");
			return -EINVAL;
		}
		const auto net_basename = net_pathname.substr(pos + 1);

		env_manager::getInstance().insert_envvar("NCCL_TUNER_PLUGIN", net_basename, false);
	} else {
		NCCL_OFI_WARN("Failed to find shared library info. Not configuring NCCL_TUNER_PLUGIN.");
		return -EINVAL;
	}

	return 0;
}




#endif /* HAVE_CUDA */

/*
 * @brief	Update NCCL's system topology using static pre-configured topology
 * 		files for supported EC2 platform types.
 *
 * @return	0, when we are succesfully able to update NCCL topology or
 * 		   if we find no match
 * 		error, on failure
 */
int platform_init(const char **provider_filter)
{
	int ret = ncclSuccess;
	struct ec2_platform_data *platform_data;
	bool select_efa = false;
	char *fi_provider;

	NCCL_OFI_INFO(NCCL_INIT, "Configuring AWS-specific options");

	platform_data = get_platform_data();

	/* if we're here, we think we're on an EC2 instance, so force
	 * EFA provider (for platforms without EFA, this will cause a
	 * fallback to NCCL's internal TCP.  In the case of Neuron, a
	 * hard failure when there are no NICs.  Both are the
	 * behaviors we want).
	 */
	fi_provider = getenv("FI_PROVIDER");
	if (fi_provider == NULL) {
		NCCL_OFI_INFO(NCCL_INIT, "Setting provider_filter to efa");
		*provider_filter = "efa";
		select_efa = true;
	} else if (0 == strcmp(fi_provider, "efa")) {
		select_efa = true;
	}

	if (platform_data != NULL) {
		env_manager::getInstance().insert_envvars(platform_data->env);
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
	uint32_t libversion = fi_version();
	const char * fork_safe_var_name =
		(FI_MAJOR(libversion) > 1 || (FI_MAJOR(libversion) == 1 && FI_MINOR(libversion) >= 13))
		? "FI_EFA_FORK_SAFE"
		: "RDMAV_FORK_SAFE";
	env_manager::getInstance().insert_envvar(fork_safe_var_name, "1", false);

	ret = configure_nvls_option();
	if (ret != 0) {
		NCCL_OFI_WARN("Unable to configure NVLS option");
		goto exit;
	}

	ret = configure_tuner();
	if (ret != 0) {
		NCCL_OFI_WARN("Unable to configure tuner: %s", strerror(-ret));
		goto exit;
	}
#endif

	/*
	 * Update topology if platform topology is available and
	 * environment variable NCCL_TOPO_FILE is not set.
	 */
	if (getenv("NCCL_TOPO_FILE")) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Running on %s platform, NCCL_TOPO_FILE environment variable is already set to %s",
			      nccl_net_ofi_get_product_name(), getenv("NCCL_TOPO_FILE"));
	} else if (platform_data && platform_data->topology) {
		char topology_path[PATH_MAX];

		ret = snprintf(topology_path, sizeof(topology_path), "%s/%s",
			       XML_DIR, platform_data->topology);
		if (ret < 0 || (size_t)ret >= sizeof(topology_path)) {
			NCCL_OFI_WARN("Error occurred while forming the complete topology XML file path. RC: %d, Buffer Size: %d, XML dir: %s, Topology file: %s",
				      ret, PATH_MAX, XML_DIR, platform_data->topology);
			ret = -ENOMEM;
			goto exit;
		}
		ret = 0;

		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
				"Running on %s platform, topology file %s",
				nccl_net_ofi_get_product_name(), topology_path);

		env_manager::getInstance().insert_envvar("NCCL_TOPO_FILE", topology_path, false);
	}

	if (nic_dup_conns == 0 && platform_data)
		nic_dup_conns = platform_data->default_dup_conns;

	if (ofi_nccl_net_latency.get_source() == ParamSource::DEFAULT) {
		if (platform_data && platform_data->latency >= 0.0) {
			ofi_nccl_net_latency.set(platform_data->latency);
		} else {
			/*
			 * Empirical testing on P5 had shown that NCCL's
			 * internal tuner choices work better with this value.
			 * While this needs to be revisited for newer
			 * generations of EFA, using it as the fall-through
			 * default for undefined platforms.
			 */
			ofi_nccl_net_latency.set(75.0);
		}
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Internode latency set at %.1f us",
			      ofi_nccl_net_latency.get());
	}

	if (select_efa && ofi_nccl_protocol.get_source() == ParamSource::DEFAULT && platform_data) {
		ofi_nccl_protocol.set(platform_data->default_protocol);
	}

exit:
	return ret;
}

int platform_config_endpoint(struct fi_info *info, struct fid_ep* endpoint) {
	int ret = 0;
#if HAVE_CUDA
	const char *optname_name = "none";
	int optname = -1;
#endif

	if (endpoint == NULL) {
		NCCL_OFI_WARN("Unable to configure invalid endpoint");
		ret = -EINVAL;
		goto exit;
	}

	/* short circuit when not using EFA */
	if (0 != strcmp(info->fabric_attr->prov_name, "efa")) {
		ret = 0;
		goto exit;
	}

	if (ofi_nccl_disable_gdr_required_check() == 0) {
		/* Ensure GDR is enabled on GDR-supported instances */
		struct ec2_platform_data *platform_data = get_platform_data();
		if (platform_data && platform_data->gdr_required && support_gdr != GDR_SUPPORTED) {
			NCCL_OFI_WARN("GDR disabled on GDR-supported instance type %s", platform_data->name);
			ret = -EINVAL;
			goto exit;
		}
	}

	/* If the selected communication protocol is RDMA write and the user did
	 * not disable the native RDMA support check, validate that the
	 * FI_OPT_EFA_EMULATED_WRITE endpoint option can be accessed, and that
	 * emulated writes are disabled.
	 */

	if (0 == strcasecmp("RDMA", ofi_nccl_protocol.get()) &&
	    ofi_nccl_disable_native_rdma_check() == 0) {
		ret = validate_rdma_write(endpoint);
		if (ret != 0) {
			goto exit;
		}
	}

#if HAVE_CUDA
	static bool nccl_proto_configured = false;
	static bool need_ordering = false;
	static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

	/* During initialization, try to set
	 * FI_OPT_EFA_{SENDRECV,WRTIE}_IN_ORDER_ALIGNED_128_BYTES to
	 * true to see if the LL/LL128 protocol is supported. After
	 * initialization, try to set the option to true again if it
	 * was previously set and error if we can't set them the same
	 * way later.
	 */
	if (0 == strcasecmp("SENDRECV", ofi_nccl_protocol.get())) {
#if HAVE_DECL_FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES
		optname = FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES;
		optname_name = "FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES";
#endif
	} else if (0 == strcasecmp("RDMA", ofi_nccl_protocol.get())) {
#if HAVE_DECL_FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES
		optname = FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES;
		optname_name = "FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES";
#endif
	} else {
		NCCL_OFI_WARN("unkonwn transport %s", ofi_nccl_protocol.get());
		ret = -EINVAL;
		goto exit;
	}

	nccl_net_ofi_mutex_lock(&mutex);

	/* TODO: This is a temporary hack to disable setting
	 * NCCL_PROTO=simple on P5en when using the RDMA protocol.  EFA
	 * on P5en does not currently report
	 * WRITE_IN_ORDER_ALIGNED_128_BYTES because it can deliver the
	 * (correct) payload twice.  This violates the meaning of the
	 * WRITE_IN_ORDER_ALIGNED_128_BYTES flag in rdma-core, but
	 * does not violate any assumptions about buffer reuse in
	 * NCCL. We have confirmed that the EFA provider in Libfabric
	 * will not segment messages for fi_write(), so this is safe.
	 * Note that the SENDRECV protocol does have segmentation
	 * challenges that require us to obey the
	 * SENDRECV_IN_ORDER_ALIGNED_128_BYTES flag, so we only skip
	 * the check when using the RDMA protocol.
	 */
	if (!nccl_proto_configured) {
		if ((NULL == getenv("NCCL_PROTO")) &&
		    (0 == strcasecmp("RDMA", ofi_nccl_protocol.get())) &&
		    (0 == strcmp(nccl_net_ofi_get_product_name(), "p5en.48xlarge"))) {
			NCCL_OFI_INFO(NCCL_INIT, "Skipping NCCL_PROTO checks on P5en + RDMA");
			need_ordering = false;
			nccl_proto_configured = true;
		}
	}

	/* If we know we need byte delivery ordering (need_ordering ==
	 * true) or this is the first time that we're configuring an
	 * endpoint (nccl_proto_configured == false), then try to
	 * configure ordering on the endpoint.  The only time we care
	 * about ordering is if we don't set NCCL_PROTO=simple,
	 * because previous endpoints were able to be configured with
	 * ordering.  If we're not expecting ordering, we don't really
	 * care if ordering is on or off for the endpoint.
	 */
	if (need_ordering || !nccl_proto_configured) {
		bool have_ordering = false;

		if (optname != -1) {
			ret = configure_ep_inorder(endpoint, optname, optname_name,
						   &have_ordering);
			if (ret != 0) {
				NCCL_OFI_WARN("Unexpected failure setting inorder %d", ret);
				goto unlock;
			}
		}

		if (need_ordering && !have_ordering) {
			NCCL_OFI_WARN("Setting %s option failed after succeeding during initialization",
				      optname_name);
			ret = -ENOTSUP;
			goto unlock;
		}

		if (!nccl_proto_configured) {
			need_ordering = have_ordering;
			nccl_proto_configured = true;

			if (!have_ordering) {
				/* When byte delivery ordering is not guaranteed, force
				 * the simple protocol as the LL/LL128 protocols can lead
				 * to data corruption without data delivery ordering.
				 */
				ret = nccl_net_ofi_configure_nccl_proto_simple("byte delivery ordering");
				if (ret != 0) {
					NCCL_OFI_WARN("Failed to set NCCL_PROTO: %d", ret);
					ret = -ENOTSUP;
					goto unlock;
				}
			}
		}
	}

	if (0 == strcasecmp("RDMA", ofi_nccl_protocol.get())) {
		ret = configure_ep_max_msg_size(endpoint);
		if (ret != 0) {
			NCCL_OFI_WARN("Unexpected failure setting max_msg_size %d", ret);
			goto unlock;
		}
	}

unlock:
	nccl_net_ofi_mutex_unlock(&mutex);
#endif // HAVE_CUDA

exit:
	return ret;
}

static const struct platform_aws_node_guid* get_node_guid_fields(struct fi_info *info)
{
	if (!info->nic || !info->nic->device_attr || !info->nic->device_attr->name) {
		NCCL_OFI_WARN("fi_nic attributes not available.");
		return nullptr;
	}

	std::string device_name = info->nic->device_attr->name;

	/* Check to see if we've already parsed the fields for this RDMA device */
	{
		std::lock_guard<std::mutex> lock(cache_mutex);
		auto it = guid_cache.find(device_name);
		if (it != guid_cache.end()) {
			return &(it->second);
		}
	}

	std::string filepath = "/sys/class/infiniband/";
	filepath += info->nic->device_attr->name;
	filepath += "/node_guid";

	std::ifstream file(filepath);
	if (!file.is_open()) {
		throw std::system_error(errno, std::system_category(),
			"Failed to open " + filepath);
	}

	std::string guid_str;
	if (!std::getline(file, guid_str)) {
		throw std::runtime_error("Failed to read data from " + filepath);
	}

	/* Parse the GUID string in XXXX:XXXX:XXXX:XXXX format */
	unsigned int a, b, c, d;
	if (sscanf(guid_str.c_str(), "%4x:%4x:%4x:%4x", &a, &b, &c, &d) != 4) {
		throw std::runtime_error("Invalid GUID format in " + filepath);
	}

	/* Reconstruct the 64-bit value */
	uint64_t raw_value = ((uint64_t)a << 48) |
			    ((uint64_t)b << 32) |
			    ((uint64_t)c << 16) |
			    ((uint64_t)d);

	NCCL_OFI_INFO(NCCL_INIT, "GUID of %s: %016lx", info->nic->device_attr->name, raw_value);

	/*
	 * +--------------------+---------------------+------------------+------------+
	 * |63                32|31                 16|15               8|7          0|
	 * +--------------------+---------------------+------------------+------------+
	 * | func_mac_low_bytes | per_card_pci_domain | per_card_pci_bus |  func_idx  |
	 * +--------------------+---------------------+------------------+------------+
	 */
	struct platform_aws_node_guid node_guid_fields;
	node_guid_fields.func_idx = raw_value & 0xFF;
	node_guid_fields.per_card_pci_bus = (raw_value >> 8) & 0xFF;
	node_guid_fields.per_card_pci_domain = (raw_value >> 16) & 0xFF;
	node_guid_fields.func_mac_low_bytes = (raw_value >> 32);

	/* Stash in the guid fields cache */
	{
		std::lock_guard<std::mutex> lock(cache_mutex);
		guid_cache[device_name] = node_guid_fields;
		return &(guid_cache[device_name]);
	}
}

static int get_rail_vf_idx(struct fi_info *info)
{
	const struct platform_aws_node_guid* fields = get_node_guid_fields(info);
	if (fields == nullptr) {
		NCCL_OFI_WARN("Failed to get node GUID fields");
		return -EIO;
	}
	return fields->func_idx;
}

void platform_device_set_guid(struct fi_info *info, struct nccl_net_ofi_device *device)
{
	const struct platform_aws_node_guid* fields = get_node_guid_fields(info);
	uint32_t node_id = nccl_ofi_get_unique_node_id();

	if (!fields ||
	    strcmp("0xefa0", info->nic->device_attr->device_id) == 0 ||
	    strcmp("0xefa1", info->nic->device_attr->device_id) == 0 ||
	    strcmp("0xefa2", info->nic->device_attr->device_id) == 0) {

		device->guid = (static_cast<uint64_t>(node_id) << 32) | device->dev_id;
	} else {
		device->guid = (static_cast<uint64_t>(node_id) << 32) |
			       (fields->per_card_pci_domain << 8) |
				fields->per_card_pci_bus;
	}

	NCCL_OFI_INFO(NCCL_INIT, "GUID for dev[%d]: %032lx", device->dev_id, device->guid);
}

/*
 * On P5/P5e, there are up to 32 EFA devices.  Each pair of EFA
 * devices shares some Nitro card resources, and there is a marginal
 * performance gain if the 0th device in the pair only talks to 0th
 * devices in the remote and so on.  Unfortunately, the hypervisor is
 * not consistent in mapping BDFs between the two devices that share
 * resources such that the natural Libfabric provider sorting ends up
 * with this pairing happening naturally.  So this code reorders the
 * provider list to make that happen.
 *
 * We maintain BDF ordering in general, but do minimal reordering so
 * that there is an alternating of the 0th pair index and then the 1st
 * pair index, and so on.
 */
void platform_sort_rails(struct fi_info **info_list, size_t num_rails, size_t num_groups)
{
	struct fi_info **info_array = NULL;
	struct fi_info *info_iter = NULL;
	size_t *vf_array = NULL;
	struct fi_info *output_info_list = NULL;
	struct fi_info *output_info_end = NULL;
	size_t highest_vf_idx = 0;
	size_t next_vf_idx = 0;
	size_t info_count;

	/* we only want to reorder if there's more than one NIC per
	 * group (ie, per GPU).  Less than that (P4d or trainium), we
	 * assume topo ordering is sufficient */
	if ((num_rails / num_groups) <= 1) {
		return;
	}

	info_array = (struct fi_info **)calloc(num_rails, sizeof(struct fi_info*));
	if (info_array == NULL) {
		NCCL_OFI_WARN("Did not reorder arrays due to calloc failure");
		goto cleanup;
	}

	vf_array = (size_t *)calloc(num_rails, sizeof(size_t));
	if (vf_array == NULL) {
		NCCL_OFI_WARN("Did not reorder arrays due to calloc failure");
		goto cleanup;
	}

	/* copy the input list into an array so that we can more *
	 * easily associate more data (like the vf array) with the
	 * input and keep everything organized */
	info_iter = *info_list;
	info_count = 0;
	while (info_iter != NULL && info_count < num_rails) {
		info_array[info_count] = fi_dupinfo(info_iter);
		if (info_array[info_count] == NULL) {
			NCCL_OFI_WARN("fi_dupinfo failed");
			goto cleanup;
		}
		info_iter = info_iter->next;

		int ret = get_rail_vf_idx(info_array[info_count]);
		if (ret < 0) {
			NCCL_OFI_WARN("lookup of rail for index %lu failed: %s",
				      info_count, strerror(-ret));
			goto cleanup;
		}
		vf_array[info_count] = ret;
		if (vf_array[info_count] > highest_vf_idx) {
			highest_vf_idx = vf_array[info_count];
		}

		info_count++;
	}
	if (info_count != num_rails) {
		NCCL_OFI_WARN("Info count (%lu) and num_rails (%lu) do not match.  Aborting reorder.",
			      info_count, num_rails);
		goto cleanup;
	}

	/* No reorder required, as devices all have the same vf idx
	   and end result would be the input array */
	if (highest_vf_idx == 0) {
		goto cleanup;
	}

	for (size_t i = 0 ; i < num_rails ; i++) {
		size_t j = num_rails;
		for (j = 0 ; j < num_rails ; j++) {
			if (info_array[j] == NULL) {
				continue;
			}

			if (vf_array[j] == next_vf_idx) {
				if (output_info_list == NULL) {
					output_info_list = output_info_end = info_array[j];
				} else {
					output_info_end->next = info_array[j];
					output_info_end = info_array[j];
				}
				info_array[j] = NULL;
				next_vf_idx = (next_vf_idx + 1) % (highest_vf_idx + 1);
				break;
			}
		}
		if (j == num_rails) {
			NCCL_OFI_WARN("Did not find a device with expected index %zu", next_vf_idx);
			goto cleanup;
		}
	}

	fi_freeinfo(*info_list);
	*info_list = output_info_list;
	output_info_list = NULL;

cleanup:
	if (info_array != NULL) {
		for (size_t i = 0 ; i < num_rails ; i++) {
			if (info_array[i] != NULL) {
				fi_freeinfo(info_array[i]);
			}
		}
		free(info_array);
	}
	if (vf_array != NULL) {
		free(vf_array);
	}
	if (output_info_list != NULL) {
		fi_freeinfo(output_info_list);
	}

	return;
}


bool platform_default_domain_per_thread(void)
{
	struct ec2_platform_data *platform_data = get_platform_data();
	if (platform_data != NULL && platform_data->domain_per_thread != 0) {
		return true;
	}
	return false;
}
