/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <ctype.h>

#include "nccl_ofi.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_tracepoint.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif
#include "nccl_ofi_sendrecv.h"
#include "nccl_ofi_rdma.h"
#include "nccl_ofi_topo.h"
#include "nccl_ofi_math.h"
#include "nccl_ofi_idpool.h"
#include "nccl_ofi_dmabuf.h"
#include "nccl_ofi_platform.h"
#include "nccl_ofi_ofiutils.h"
#include "nccl_ofi_system.h"

/* Indicates if GPUDirect is supported by libfabric provider */
enum gdr_support_level_t support_gdr = GDR_UNKNOWN;

/* Indicates if the cudaDeviceFlushGPUDirectRDMAWrites function should be used
 * to flush data to the GPU. Note, CUDA flush support is not supported on all
 * platforms and should be disabled by default */
bool cuda_flush = false;

/* number of duplicate providers to create for each discovered
 * provider, including renaming to cause NCCL to create additional
 * rings to use the connections
 */
int nic_dup_conns = 0;

/* number of cq entries to read in a single call to fi_cq_read.
   This variable will be updated during init (hence, can not be
   const), but will not change during execution.  Therefore, it may be
   read in the polling loop without protection of a lock. */
size_t cq_read_count = 1;

/* Indicates if memory registration of local buffers is required */
bool local_mr = false;
/* Indicates if endpoint memory registration is required */
bool endpoint_mr = false;

/* Indicates if remote virtual addressing is used */
bool virt_addr_mr = false;

/* Selected communication protocol. */
const char *nccl_ofi_selected_protocol = NULL;

/* Internode network latency. */
float net_latency = .0;

/* Size of a memory page */
size_t system_page_size = 0;

/* Alignment used for MR cache and key creation */
size_t mr_cache_alignment = 0;

/*
 * @brief	Allocate memory region for memory registration
 *
 * This function allocates memory that covers full page aligned.
 *
 * Internally allocated memory that is registered is required to cover
 * full memory pages. For more information, see functions
 * `register_internal_mr_buffers()` and `reg_internal_mr_ep()`.
 *
 * To free deallocate the memory region, function
 * nccl_net_ofi_dealloc_mr_buffer() must be used.
 *
 * @param	size
 *		Size of the memory region. Must be a multiple of system memory page size.
 * @return	Pointer to memory region. Memory region is aligned to system memory page size.
 * @return	0, on success
 *		error, on others
 */
int nccl_net_ofi_alloc_mr_buffer(size_t size, void **ptr)
{
	assert(system_page_size > 0);
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	*ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
		    MAP_PRIVATE | MAP_ANON, -1, 0);
	if (OFI_UNLIKELY(*ptr == MAP_FAILED)) {
		NCCL_OFI_WARN("Unable to map MR buffer (%d %s)",
			      errno, strerror(errno));
		*ptr = NULL;
		return -errno;
	}
	assert(NCCL_OFI_IS_PTR_ALIGNED(*ptr, system_page_size));
	return 0;
}

/*
 * @brief	Deallocate memory region allocated by function nccl_net_ofi_alloc_mr_buffer()
 *
 * @return	Pointer to memory region
 * @param	size
 *		Size of the memory region
 * @return	0, on success
 *		error, on others
 */
int nccl_net_ofi_dealloc_mr_buffer(void *ptr, size_t size)
{
	int ret = 0;

	assert(NCCL_OFI_IS_PTR_ALIGNED(ptr, system_page_size));
	assert(NCCL_OFI_IS_ALIGNED(size, system_page_size));

	ret = munmap(ptr, size);
	if (OFI_UNLIKELY(ret != 0)) {
		NCCL_OFI_WARN("Unable to unmap MR buffer (%d %s)",
			      errno, strerror(errno));
		ret = -errno;
	}
	return ret;
}


int nccl_net_ofi_create_plugin(nccl_net_ofi_plugin_t **plugin_p)
{
	int ret = 0;
	const char *provider_filter = NULL;
	nccl_net_ofi_plugin_t *plugin;
	nccl_net_ofi_ep_t *base_ep = NULL;
	nccl_net_ofi_device_t *device = NULL;
	nccl_ofi_properties_t properties;

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Initializing " PACKAGE_STRING);

	/* Print Libfabric version */
	uint32_t fab_version = fi_version();
	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using Libfabric version %u.%u", FI_MAJOR(fab_version),
			FI_MINOR(fab_version));

	long int system_page_size_sysconf = sysconf(_SC_PAGESIZE);
	if (OFI_UNLIKELY(system_page_size_sysconf == -1)) {
		NCCL_OFI_WARN("Failed to get system page size (%d %s)", errno, strerror(errno));
		ret = -ENOTSUP;
		goto exit;
	}
	system_page_size = (size_t)system_page_size_sysconf;
	assert(NCCL_OFI_IS_POWER_OF_TWO(system_page_size));
	assert(system_page_size > 0);
	/*
	 * System page size isn't reflective of the GDR mappings. We're not trying to map a
	 * whole page, but just to find an interval that makes an array-based cache manageable.
	 */
	mr_cache_alignment = NCCL_OFI_MIN(system_page_size, NCCL_OFI_CACHE_PAGE_SIZE);

#if HAVE_CUDA
	ret = nccl_net_ofi_cuda_init();
	if (ret != 0) {
		NCCL_OFI_WARN("CUDA initialization failed.");
		goto exit;
	}
#endif

	/* configuration parameters */
	nic_dup_conns = ofi_nccl_nic_dup_conns();
	net_latency = (float)ofi_nccl_net_latency();
	cq_read_count = ofi_nccl_cq_read_count();

	if (platform_init) {
		ret = platform_init(&provider_filter);
		if (ret != 0)
			goto exit;
	}

	/* This is ugly, but here's the basic protocol selection
	 * logic:
	 *   1. if the user set NCCL_OFI_PROTOCOL, use that.
	 *   2. if the platform init set nccl_ofi_selected_protocol,
	 *      use that.
	 *   3. If the rdma protocol reports multiple nics per device
	 *      and initialized successfully, use that.
	 *   4. If the sendrecv protocol initialized successfully, use
	 *      that
	 *   5. If the rdma protocol initialized successfully, use
	 *      that.
	 */
	if (ofi_nccl_protocol()) {
		nccl_ofi_selected_protocol = ofi_nccl_protocol();
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using transport protocol %s (user set)",
			      nccl_ofi_selected_protocol);
	} else if (nccl_ofi_selected_protocol != NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using transport protocol %s (platform set)",
			      nccl_ofi_selected_protocol);
	}

	if (nccl_ofi_selected_protocol != NULL) {
		bool dummy;

		if (0 == strcasecmp(nccl_ofi_selected_protocol, "SENDRECV")) {
			ret = nccl_net_ofi_sendrecv_init(provider_filter, &plugin);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to initialize sendrecv protocol");
				goto exit;
			}
		} else if (0 == strcasecmp(nccl_ofi_selected_protocol, "RDMA")) {
			ret = nccl_net_ofi_rdma_init(provider_filter, &plugin, &dummy);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to initialize rdma protocol");
				goto exit;
			}
		} else {
			NCCL_OFI_WARN("Unable to find plugin protocol %s", nccl_ofi_selected_protocol);
			ret = -ENOTSUP;
			goto exit;
		}
	} else {
		bool have_multiple_rails = false;
		nccl_net_ofi_plugin_t *rdma_plugin = NULL, *sendrecv_plugin = NULL;

		ret = nccl_net_ofi_rdma_init(provider_filter, &rdma_plugin, &have_multiple_rails);
		if (ret != 0) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Failed to initialize rdma protocol: %s", fi_strerror(-ret));
			have_multiple_rails = false;
			rdma_plugin = NULL;
		}

		if (!have_multiple_rails || rdma_plugin == NULL) {
			ret = nccl_net_ofi_sendrecv_init(provider_filter, &sendrecv_plugin);
			if (ret != 0) {
				NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
					       "Failed to initialized sendrecv protocol: %s", fi_strerror(-ret));
				sendrecv_plugin = NULL;
			}
		}

		if (have_multiple_rails && rdma_plugin != NULL) {
			nccl_ofi_selected_protocol = "RDMA";
			plugin = rdma_plugin;
			if (sendrecv_plugin != NULL) {
				sendrecv_plugin->release_plugin(sendrecv_plugin);
			}
		} else {
			nccl_ofi_selected_protocol = "SENDRECV";
			plugin = sendrecv_plugin;
			if (rdma_plugin != NULL) {
				rdma_plugin->release_plugin(rdma_plugin);
			}
		}

		if (nccl_ofi_selected_protocol == NULL || plugin == NULL) {
			NCCL_OFI_WARN("Unable to find a protocol that worked.  Failing initialization.");
			ret = -EINVAL;
			goto exit;
		}

		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using transport protocol %s",
			      nccl_ofi_selected_protocol);
	}

	if (ofi_nccl_domain_per_thread() != -1) {
		plugin->domain_per_thread = (ofi_nccl_domain_per_thread() > 0);
	} else {
		if (platform_default_domain_per_thread) {
			plugin->domain_per_thread = platform_default_domain_per_thread();
		} else {
			plugin->domain_per_thread = false;
		}
	}
	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Creating one domain per %s",
		      plugin->domain_per_thread ? "thread" : "process");

	ret = plugin->complete_init(plugin);
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to initialize %s protocol", nccl_ofi_selected_protocol);
		goto exit;
	}

	/* In order to set endpoint options and potentially NCCL configuration
	 * options (such as NCCL_PROTO) during the plugin initialization
	 * process, we need to create an endpoint and call the platform hook
	 * "platform_config_endpoint" using "get_ep". This code makes the
	 * assumption that the thread calling "nccl_net_ofi_init" will make
	 * communication calls. As well, since without this code the endpoint
	 * would be created the first time "get_ep" in called during a listen or
	 * connect call, creating the endpoint earlier would not be a waste of
	 * resources. This initialization happens once per process, and thus it
	 * does not matter which device is used to create the endpoint.
	 */
	device = plugin->get_device(plugin, 0);

	ret = device->get_ep(device, &base_ep);
	if (ret != 0) {
		goto exit;
	}
	ret = device->get_properties(device, &properties);
	if (ret != 0) {
		goto exit;
	}
	NCCL_OFI_INFO(NCCL_NET | NCCL_INIT, "Support for global registrations: %s",
		      (properties.regIsGlobal == 0) ? "false" : "true");
	NCCL_OFI_INFO(NCCL_NET | NCCL_INIT, "Support for DMA-BUF registrations: %s",
		      (properties.dmabuf_support == 0) ? "false" : "true");
	/* Cause release to not actually free the resources, to speed
	 * up initialization, since the very same resources will be
	 * recreated by NCCL soon after initialization to do real
	 * communication.
	 */
	base_ep->ref_cnt++;
	ret = base_ep->release_ep(base_ep);
	base_ep->ref_cnt--;
	if (ret != 0) {
		goto exit;
	}

	assert(support_gdr != GDR_UNKNOWN);

	/* we don't actually know if GDR is supported until we've
	 * created the first endpoint, so this check needs to be way
	 * down here
	 */
	if (nic_dup_conns > 0 && support_gdr != GDR_UNSUPPORTED) {
		NCCL_OFI_WARN("NCCL_OFI_NIC_DUP_CONNS set on platform that supports GPUDirect RDMA.  This configuration is not supported.");
		ret = -ENOTSUP;
		goto exit;
	}

	*plugin_p = plugin;

 exit:
	if (ret != 0) {
		NCCL_OFI_WARN(PACKAGE_NAME " initialization failed");
	}
	return ret;
}

static int get_device_pci_path(struct fid_nic *nic_info, char** path)
{
	int ret = 0;
	struct fi_pci_attr *pci = NULL;
	char *device_path = NULL;

	if (nic_info->bus_attr->bus_type != FI_BUS_PCI) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "Invalid type of PCI bus returned %d",
			      nic_info->bus_attr->bus_type);
		ret = -EINVAL;;
		goto exit;
	}

	pci = &nic_info->bus_attr->attr.pci;
	ret = asprintf(&device_path,
		       "/sys/class/pci_bus/%04x:%02x/../../%04x:%02x:%02x.%01x",
		       pci->domain_id, pci->bus_id,
		       pci->domain_id, pci->bus_id, pci->device_id, pci->function_id);
	if (ret < 0) {
		NCCL_OFI_WARN("pciPath: Allocation failure");
		ret = -ENOMEM;
		goto exit;
	} else {
		ret = 0;
	}

	*path = realpath(device_path, NULL);
	if (*path == NULL) {
		NCCL_OFI_WARN("pciPath: Could not find real path of %s",
			      device_path);
		ret = -errno;
		goto exit;
	}

 exit:
	if (device_path)
		free(device_path);

	return ret;
}

/*
 * @brief	Set default properties for libfabric NIC info.
 */
static int set_nic_props_default(int dev_id, struct fi_info *nic_prov,
				 nccl_ofi_properties_t *props)
{
	props->name = strdup(nic_prov->domain_attr->name);

	/*
	 * Currently, libfabric providers provide multiple `fi_info`
	 * objects for devices with multiple ports. So, safely assume port number
	 * to be always 1.
	 */
	props->port_number = 1;
	props->max_communicators = 0;
	props->guid = dev_id;

	props->latency = net_latency >= .0 ? net_latency : .0;

	/*
	 * Maximum number of grouped receives. Currently, we set it to 1 to
	 * maintain single send/recv semantics (similar to NCCL versions < v2.12).
	 *
	 * Grouped receives are useful for alltoall collectives where one
	 * receiver is expected to receive from multiple remote GPUs using
	 * PXN(PCIe X NVLINK) feature. Other collectives like allreduce aren't
	 * impacted with this feature as NCCL doesn't aggregate receives from
	 * same source.
	 */
	props->max_group_receives = NCCL_OFI_MAX_RECVS;

	if (support_gdr == GDR_SUPPORTED) {
		props->hmem_support = true;
	} else {
		props->hmem_support = false;
	}

	props->dmabuf_support = false;

	/* Should be successful for ptrSupport invocation */
	return 0;
}

/*
 * @brief	Set properties obtained from libfabric NIC Info.
 *
 * @return	Populated props structure
 */
int nccl_net_ofi_info_properties(nccl_net_ofi_plugin_t *plugin, struct fi_info *nic_prov,
				 int dev_id, int num_devices, nccl_ofi_properties_t *props)
{
	int ret = 0;
	struct fid_nic *nic_info = NULL;
	const char *platform_type = NULL;

	memset(props, 0, sizeof(*props));

	ret = set_nic_props_default(dev_id, nic_prov, props);
	if (ret != 0) {
		goto error;
	}

	/* Change default values as set by NIC attributes */
	nic_info = (struct fid_nic *)nic_prov->nic;
	if (nic_info == NULL) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET,
			      "No NIC info for dev %d. Supplying default values for NIC properties.",
			      dev_id);
		ret = 0;
		goto exit;
	}

	/* name is NULL if device is a part of multirail config */
	/* overriding default name only if value is available from provider */
	if (nic_info->device_attr->name) {
		if (props->name) {
			free(props->name);
		}
		props->name = strdup(nic_info->device_attr->name);
		assert(props->name != NULL);
	}

	/*
	 * Determine the scope of MRs for providers to report global registration
	 * support to NCCL.
	 * NCCL uses regIsGlobal to determine support for User Registrations via
	 * the NCCL API. If providers tie MRs to endpoints, the plugin can not
	 * support this model (since NCCL maintains a per-domain registration
	 * cache which requires (domain-)global registrations.
	 * Also, if we have different domains for different threads, registrations
	 * are not reported as global even if they are tied to the domain.
	 */
	if (nic_prov->domain_attr->mr_mode & FI_MR_ENDPOINT || plugin->domain_per_thread) {
		props->regIsGlobal = 0;
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Global registrations are not supported");
	} else {
		props->regIsGlobal = 1;
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Global registrations supported");
	}

	/* Speed reported in Mbps */
	props->port_speed = nic_info->link_attr->speed / (1e6);

	/*
	 * When running on AWS, newer platforms might return incorrect link
	 * speeds when running a version of the driver that does not contain
	 * this change to query the device:
	 * https://github.com/amzn/amzn-drivers/commit/c4c7926561741c97f78e27836f5687bf16c54b23
	 * AND running a version of libfabric that does not contain this change:
	 * https://github.com/ofiwg/libfabric/pull/10496/commits/fd0c5f0b0abe91fc062ad57834a93f35278d2392
	 *
	 * Until these updates are more widely deployed, the following override
	 * fixes port_speed for impacted platforms.
	 */
	platform_type = nccl_net_ofi_get_product_name();
	if (platform_type != NULL && strcmp(platform_type, "p5en.48xlarge") == 0) {
		NCCL_OFI_TRACE(NCCL_INIT, "Overriding OFI link_attr speed to 200Gbps/link for P5en platform");
		props->port_speed = 200 * (1e3);
	}

	ret = get_device_pci_path(nic_info, &props->pci_path);
	if (ret != 0) {
		ret = 0;
		props->pci_path = NULL;
	}

	if (nic_dup_conns > 1) {
#if HAVE_CUDA
		int num_gpus_visible = nccl_net_ofi_cuda_get_num_devices();
		int active_cuda_device = nccl_net_ofi_cuda_get_active_device_idx();
		int gpus_per_conn = -1;
		int c = 0;

		if (!(num_gpus_visible > 0)) {
			NCCL_OFI_WARN("Error getting CUDA device count");
			ret = -ENOTSUP;
			goto error;
		}

		if (active_cuda_device < 0 || active_cuda_device >= num_gpus_visible) {
			NCCL_OFI_WARN("Error getting current CUDA device");
			ret = -ENOTSUP;
			goto error;
		}

		gpus_per_conn = num_gpus_visible / num_devices;
		if (gpus_per_conn == 0) gpus_per_conn = 1;

		/* The goal is to have gpus_per_conn gpus in the local
		 * system think that any given virtual nic is the one
		 * that they should use, and that it is different than
		 * the other NICs in the system.  We do this by only
		 * leaving a valid device id in pci_path when
		 * active_cuda_device / gpus_per_comm is equal to the
		 * NIC dev index we're currently querying.  For the
		 * others, we provide a PCIPath that points at the PCI
		 * Bus itself, which NCCL will interpret to be on a
		 * different complex than the bus (assuming the NIC
		 * BUS and GPU BUS are the same).
		 *
		 * There are a bunch of assumptions in this logic,
		 * such that the physical NICs in the system don't
		 * have PCI affinity with the GPUs.  Given that we've
		 * already established that GPUDirect doesn't work,
		 * this is probably ok; any affinity is lost by
		 * bouncing through host buffers anyway.
		 */
		if ((active_cuda_device / gpus_per_conn != dev_id) && props->pci_path) {
			for (c = strlen(props->pci_path); props->pci_path[c] != '/'; c--) {
				props->pci_path[c] = '\0';
			}
		}
		NCCL_OFI_TRACE(NCCL_INIT,
			       "Returning synthetic PCI path for device %d of %s",
			       dev_id,
			       props->pci_path);

		snprintf(props->name,
			 FI_NAME_MAX + 2,
			 "%s-%x",
			 nic_info->device_attr->name,
			 dev_id);
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Adjusted dev %d device name to %s",
			       dev_id,
			       props->name);
#else
		NCCL_OFI_WARN("NIC_DUP_CONNS enabled on platform that does not support NIC_DUP_CONNS.  This should not happen.");
		ret = -ENOTSUP;
		goto error;
#endif
	}

	props->max_mr_key_size = nic_prov->domain_attr->mr_key_size;


	props->dmabuf_support = ((nic_prov->caps & FI_HMEM) != 0) &&
		FI_VERSION_GE(nic_prov->fabric_attr->api_version, FI_VERSION(1, 20)) &&
		nccl_ofi_dmabuf_viable()
		;
	if (props->dmabuf_support) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "DMA-BUF support is advertised in properties.");
	}

	goto exit;
error:
	if (props->pci_path) {
		free(props->pci_path);
	}
	if (props->name) {
		free(props->name);
	}

exit:
	return ret;
}

int nccl_net_ofi_query_provider_capabilities(const struct fi_info *selected_provider,
					     unsigned int num_providers)
{
	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Selected Provider is %s (found %d nics)",
		      selected_provider->fabric_attr->prov_name, num_providers);

	if (strncmp("efa", selected_provider->fabric_attr->prov_name, strlen("efa")) == 0) {
		if (FI_VERSION_LT(fi_version(), FI_VERSION(1, 22))) {
			NCCL_OFI_WARN("EFA provider requires at least libfabric version 1.22.0.");
			return -ENOTSUP;
		}
	}

	/* Check if provider requires local memory registration */
	if (selected_provider->domain_attr->mr_mode & FI_MR_LOCAL) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s requires registration of local memory buffers",
			       selected_provider->fabric_attr->prov_name);
		local_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not require registration of local memory buffers",
			       selected_provider->fabric_attr->prov_name);
		local_mr = false;
	}

	/* Check if provider uses remote virtual addressing */
	if (selected_provider->domain_attr->mr_mode & FI_MR_VIRT_ADDR) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s uses remote virtual addressing",
			       selected_provider->fabric_attr->prov_name);
		virt_addr_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not use remote virtual addressing",
			       selected_provider->fabric_attr->prov_name);
		virt_addr_mr = false;
	}

	/* Check if provider uses endpoint memory registration */
	if (selected_provider->domain_attr->mr_mode & FI_MR_ENDPOINT) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s requires endpoint memory registration",
			       selected_provider->fabric_attr->prov_name);
		endpoint_mr = true;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s does not require endpoint memory registration",
			       selected_provider->fabric_attr->prov_name);
		endpoint_mr = false;
	}

	return 0;
}


static int nccl_net_ofi_plugin_assign_device(nccl_net_ofi_plugin_t *plugin,
					     size_t device_index,
					     nccl_net_ofi_device_t *device)
{
	if (device_index >= plugin->p_num_devs) {
		return -ENOSPC;
	}

	plugin->p_devs[device_index] = device;

	return 0;
}


static nccl_net_ofi_device_t * nccl_net_ofi_plugin_get_device(nccl_net_ofi_plugin_t *plugin,
							      size_t device_index)
{
	if (device_index >= plugin->p_num_devs) {
		NCCL_OFI_WARN("Invalid device index %zu", device_index);
		return NULL;
	}

	return plugin->p_devs[device_index];
}


static size_t nccl_net_ofi_plugin_get_num_devices(nccl_net_ofi_plugin_t *plugin)
{
	return plugin->p_num_devs;
}


int nccl_net_ofi_plugin_init(nccl_net_ofi_plugin_t *plugin,
			     size_t num_devices)
{
	plugin->p_devs =
		(nccl_net_ofi_device_t **)calloc(num_devices, sizeof(nccl_net_ofi_device_t *));
	if (plugin->p_devs == NULL) {
		NCCL_OFI_WARN("Unable to allocate "
			      "nccl_net_ofi_device_t pointer array");
		return -ENOMEM;
	}

	plugin->p_num_devs = num_devices;

	plugin->assign_device = nccl_net_ofi_plugin_assign_device;
	plugin->get_device = nccl_net_ofi_plugin_get_device;
	plugin->get_num_devices = nccl_net_ofi_plugin_get_num_devices;
	plugin->release_plugin = nccl_net_ofi_plugin_fini;

	return 0;
}


int nccl_net_ofi_plugin_fini(nccl_net_ofi_plugin_t *plugin)
{
	for (size_t i = 0 ; i < plugin->p_num_devs ; i++) {
		if (plugin->p_devs[i] != NULL) {
			plugin->p_devs[i]->release(plugin->p_devs[i]);
		}
	}

	free(plugin->p_devs);
	plugin->p_num_devs = 0;

	return 0;
}


/*
 * implementation of retreiving a domain from a device.  This code
 * assumes the device lock is already held, because in the case of
 * get_domain() we only need to worry about the device lock, but in
 * the device->get_ep call, hold the lock while we're also creating
 * the ep.
 */
static nccl_net_ofi_domain_t *nccl_net_ofi_device_get_domain_impl(nccl_net_ofi_device_t *device)
{
	nccl_net_ofi_plugin_t *plugin = NULL;
	nccl_net_ofi_domain_t *domain = NULL;
	long lookup_key = 0;

	assert(device != NULL);

	plugin = device->plugin;
	assert(plugin != NULL);

	if (plugin->domain_per_thread) {
		lookup_key = nccl_net_ofi_gettid();
	}

	HASH_FIND(hh, device->domain_table, &lookup_key,
		  sizeof(domain->creating_thread_id), domain);

	if (domain == NULL) {
		domain = device->create_domain(device);
		if (domain == NULL) {
			NCCL_OFI_WARN("Initializing a new domain for device %s failed",
				      device->name);
			return NULL;
		}

		domain->creating_thread_id = lookup_key;

		HASH_ADD(hh, device->domain_table, creating_thread_id,
			 sizeof(domain->creating_thread_id), domain);

		NCCL_OFI_TRACE(NCCL_NET, "Domain %p for device #%d (%s) is created",
			       domain,
			       device->dev_id,
			       device->name);
	}

	return domain;
}


static nccl_net_ofi_domain_t *nccl_net_ofi_device_get_domain(nccl_net_ofi_device_t *device)
{
	nccl_net_ofi_domain_t *domain;

	nccl_net_ofi_mutex_lock(&device->device_lock);
	domain = nccl_net_ofi_device_get_domain_impl(device);
	nccl_net_ofi_mutex_unlock(&device->device_lock);

	return domain;
}


static int nccl_net_ofi_device_get_ep(nccl_net_ofi_device_t *device,
				      nccl_net_ofi_ep_t **ep_p)
{
	int ret;
	nccl_net_ofi_domain_t *domain = NULL;

	nccl_net_ofi_mutex_lock(&device->device_lock);

	domain = nccl_net_ofi_device_get_domain_impl(device);
	if (domain == NULL) {
		ret = -EINVAL;
		goto unlock;
	}

	ret = domain->get_ep(domain, ep_p);

unlock:
	nccl_net_ofi_mutex_unlock(&device->device_lock);

	return ret;
}


int nccl_net_ofi_device_init(nccl_net_ofi_device_t *device, nccl_net_ofi_plugin_t *plugin,
			     int device_index, struct fi_info *ofi_info)
{
	int ret = 0;

	device->plugin = plugin;
	device->dev_id = device_index;
	device->name = strdup(ofi_info->fabric_attr->prov_name);
	if (device->name == NULL) {
		NCCL_OFI_WARN("Unable to allocate device name");
		ret = -ENOMEM;
		goto exit;
	}

	device->get_properties = NULL;
	device->get_domain = nccl_net_ofi_device_get_domain;
	device->get_ep = nccl_net_ofi_device_get_ep;
	device->get_mr_key = NULL;
	device->release = nccl_net_ofi_device_fini;

	/* Intiaialize mutex for endpoint access */
	ret = nccl_net_ofi_mutex_init(&device->device_lock, NULL);
	if (ret != 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to initialize device mutex");
		return -ret;
	}

	/* Initialize mr rkey handling */
	device->need_mr_rkey_pool = true;
	ret = nccl_ofi_mr_keys_need_own_key(ofi_info, &device->need_mr_rkey_pool);
	if (ret != 0) {
		NCCL_OFI_WARN("MR key config parsing failed: %s",
			      strerror(-ret));
		return -ret;
	}

	device->create_domain = NULL;
	device->domain_table = NULL;

exit:

	return ret;
}


int nccl_net_ofi_device_fini(nccl_net_ofi_device_t *device)
{

	if (device == NULL) {
		return 0;
	}

	if (device->name != NULL) {
		free(device->name);
	}

	return 0;
}


static int nccl_net_ofi_domain_get_ep(nccl_net_ofi_domain_t *domain,
				      nccl_net_ofi_ep_t **ep_p)
{
	int ret = 0;
	long thread_id;
	nccl_net_ofi_ep_t *ep = NULL;

	nccl_net_ofi_mutex_lock(&domain->domain_lock);

	thread_id = nccl_net_ofi_gettid();
	HASH_FIND(hh, domain->endpoint_table, &thread_id,
		  sizeof(ep->creating_thread_id), ep);

	if (ep == NULL) {
		ret = domain->create_endpoint(domain, &ep);
		if (ret != 0) {
			NCCL_OFI_WARN("Creating new endpoint for domain %p failed: %s",
				      domain, fi_strerror(-ret));
			goto unlock;
		}

		ep->creating_thread_id = thread_id;

		HASH_ADD(hh, domain->endpoint_table, creating_thread_id,
			 sizeof(ep->creating_thread_id), ep);

		NCCL_OFI_TRACE(NCCL_NET, "Eendpoint %p for domain %p is created",
			       ep, domain);
	}

	ep->ref_cnt++;
	*ep_p = ep;

unlock:
	nccl_net_ofi_mutex_unlock(&domain->domain_lock);

	return ret;
}


static int nccl_net_ofi_domain_release(nccl_net_ofi_domain_t *domain)
{
	int ret = 0;
	nccl_net_ofi_device_t *device;

	assert(domain != NULL);
	device = domain->device;

	nccl_net_ofi_mutex_lock(&domain->domain_lock);

	if (HASH_COUNT(domain->endpoint_table) == 0) {
		nccl_net_ofi_mutex_lock(&device->device_lock);
		HASH_DEL(device->domain_table, domain);

		// domain->free below is going to free the domain lock
		// and we've removed the domain from the hash table,
		// so no one should have a reference to the domain at
		// this point and we can release the mutex.
		nccl_net_ofi_mutex_unlock(&domain->domain_lock);

		ret = domain->free(domain);
		nccl_net_ofi_mutex_unlock(&device->device_lock);
		if (ret != 0) {
			NCCL_OFI_WARN("Freeing domain failed: %d", ret);
			return ret;
		}
	} else {
		nccl_net_ofi_mutex_unlock(&domain->domain_lock);
	}

	return 0;
}


int nccl_net_ofi_domain_init(nccl_net_ofi_device_t *device, nccl_net_ofi_domain_t *domain)
{
	int ret;

	domain->device = device;

	ret = nccl_net_ofi_mutex_init(&domain->domain_lock, NULL);
	if (ret != 0) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Unable to initialize domain mutex");
		return -ret;
	}

	domain->get_ep = nccl_net_ofi_domain_get_ep;
	domain->release = nccl_net_ofi_domain_release;
	domain->endpoint_table = NULL;
	domain->creating_thread_id = 0;

	domain->mr_cache = NULL;
	if (!ofi_nccl_mr_cache_disable()) {
		domain->mr_cache =
			nccl_ofi_mr_cache_init(NCCL_OFI_MR_CACHE_INIT_SIZE,
					       system_page_size);
		if (!domain->mr_cache) {
			ret = -ENOMEM;
			goto exit;
		}
	}

	if (device->need_mr_rkey_pool) {
		/* The provider may return support for a larger key size. Use
		 * the size requested by the user to allow them to limit the
		 * size of the mr_keys table. */
		const size_t shift = (ofi_nccl_mr_key_size() * 8);
		const size_t size_t_bits = (sizeof(size_t) * CHAR_BIT);
		if (shift > (size_t_bits - 1)) {
			NCCL_OFI_WARN(
				"Provided mr keypool size of %lu must be less than %zu",
				ofi_nccl_mr_key_size(),
				size_t_bits);
			return -EINVAL;
		}
		ret = nccl_ofi_idpool_init(&domain->mr_rkey_pool, 1 << shift);
	} else {
		/* Mark key pool as not in use */
		ret = nccl_ofi_idpool_init(&domain->mr_rkey_pool, 0);
	}
	if (ret != 0) {
		NCCL_OFI_WARN("Creating MR id pool failed: %s",
			      strerror(-ret));
		return -ret;
	}

exit:
	return ret;
}


int nccl_net_ofi_domain_fini(nccl_net_ofi_domain_t *domain)
{
	if (domain->mr_cache != NULL) {
		nccl_ofi_mr_cache_finalize(domain->mr_cache);
	}

	nccl_ofi_idpool_fini(&domain->mr_rkey_pool);

	return 0;
}


int nccl_net_ofi_domain_release_all(nccl_net_ofi_device_t *device)
{
	int ret, first_error = 0, domain_num;

	assert(device != NULL);
	nccl_net_ofi_domain_t *domain, *domain_tmp;
	nccl_net_ofi_ep_t *ep, *ep_tmp;

	domain_num = HASH_COUNT(device->domain_table);
	assert(domain_num > 0);
	HASH_ITER(hh, device->domain_table, domain, domain_tmp) {
		/* For each domain, clean up its endpoints. */
		nccl_net_ofi_mutex_lock(&domain->domain_lock);
		if (HASH_COUNT(domain->endpoint_table) > 0) {
			/* Ignore ref_cnt of endpoints, since only called during plugin fini. */
			HASH_ITER(hh, domain->endpoint_table, ep, ep_tmp) {
				HASH_DEL(domain->endpoint_table, ep);

				if (OFI_UNLIKELY(ep->ref_cnt > 0)) {
					NCCL_OFI_INFO(NCCL_NET, "Endpoint %p still have ref count %d when released",
						      ep, ep->ref_cnt);
				}
				ret = ep->free_ep(ep);
				if (ret != 0) {
					NCCL_OFI_WARN("Freeing endpoint failed: %d", ret);
					if (first_error != 0) {
						first_error = ret;
					}
				}
			}
		}
		nccl_net_ofi_mutex_unlock(&domain->domain_lock);

		/* domain->release takes the domain and device locks,
		 * and removes the domain from domain_table. */
		ret = domain->release(domain);
		if (ret != 0) {
			NCCL_OFI_WARN("Freeing domain failed: %d", ret);
			if (first_error != 0) {
				first_error = ret;
			}
		}
	}
	domain_num = HASH_COUNT(device->domain_table);
	if (OFI_UNLIKELY(domain_num > 0)) {
		NCCL_OFI_WARN("%u domains still active after cleanup", domain_num);
		if (first_error != 0) {
			first_error = -FI_EBUSY; // Anything else than above
		}
	}

	return first_error;
}


int nccl_net_ofi_endpoint_release(nccl_net_ofi_ep_t *ep)
{
	int ret = 0;
	nccl_net_ofi_domain_t *domain;

	assert(ep != NULL);
	domain = ep->domain;

	nccl_net_ofi_mutex_lock(&domain->domain_lock);

	ep->ref_cnt--;

	if (ep->ref_cnt == 0) {
		HASH_DEL(domain->endpoint_table, ep);

		ret = ep->free_ep(ep);
		if (ret != 0) {
			NCCL_OFI_WARN("Freeing endpoint failed: %d", ret);
			goto cleanup;
		}
	}

cleanup:
	nccl_net_ofi_mutex_unlock(&domain->domain_lock);

	if (ret == 0) {
		ret = domain->release(domain);
	}

	return ret;
}


int nccl_net_ofi_endpoint_init(nccl_net_ofi_domain_t *domain,
			       nccl_net_ofi_ep_t *ep)
{
	assert(domain != NULL);
	assert(ep != NULL);

	ep->domain = domain;
	ep->release_ep = nccl_net_ofi_endpoint_release;

	ep->creating_thread_id = 0;
	ep->ref_cnt = 0;

	return 0;
}


int nccl_net_ofi_endpoint_fini(nccl_net_ofi_ep_t *ep)
{
	/* nothing to do today */
	return 0;
}


int get_inject_rma_size_opt(struct fid_ep *ofi_ep,
			    size_t *max_write_inline_size)
{
#if HAVE_DECL_FI_OPT_INJECT_RMA_SIZE
	int ret;
	size_t optlen = sizeof(size_t);
	ret = fi_getopt(&ofi_ep->fid, FI_OPT_ENDPOINT, FI_OPT_INJECT_RMA_SIZE,
			max_write_inline_size, &optlen);
	if (ret != 0 && ret != -FI_ENOPROTOOPT) {
		NCCL_OFI_WARN("Retrieving option endpoint FI_OPT_INJECT_RMA_SIZE failed. RC: %d. Error: %s",
			      ret, fi_strerror(-ret));
	}
	return ret;
#else
	return -FI_ENOPROTOOPT;
#endif
}
