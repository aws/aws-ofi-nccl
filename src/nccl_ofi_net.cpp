/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"

#include <algorithm>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <ctype.h>

#include "nccl_ofi.h"
#include "nccl_ofi_assert.h"
#include "nccl_ofi_environ.h"
#include "nccl_ofi_param.h"
#include "nccl_ofi_tracepoint.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#elif HAVE_ROCM
#include "nccl_ofi_rocm.h"
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


extern char **environ;

/* Indicates if GPUDirect is supported by libfabric provider */
enum gdr_support_level_t support_gdr = GDR_UNKNOWN;

/* Indicates if the cudaDeviceFlushGPUDirectRDMAWrites function should be used
 * to flush data to the GPU. Note, CUDA flush support is not supported on all
 * platforms and should be disabled by default */
bool cuda_flush = false;

/* number of cq entries to read in a single call to fi_cq_read.
   This variable will be updated during init (hence, can not be
   const), but will not change during execution.  Therefore, it may be
   read in the polling loop without protection of a lock. */
size_t cq_read_count = 1;

/* Indicates if endpoint memory registration is required */
bool endpoint_mr = false;

/* Indicates if remote virtual addressing is used */
bool virt_addr_mr = false;

/* Indicates if provider's data progress model is FI_PROGRESS_AUTO */
bool data_progress_auto = false;

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
	nccl_net_ofi_ep_t *ep = NULL;
	nccl_net_ofi_device_t *device = NULL;
	nccl_ofi_properties_t properties;
	nccl_ofi_topo_t *topo = nullptr;

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
	mr_cache_alignment = std::min(system_page_size, NCCL_OFI_CACHE_PAGE_SIZE);

#if HAVE_GPU
	ret = nccl_net_ofi_gpu_init();
	if (ret != 0) {
		NCCL_OFI_WARN("CUDA initialization failed.");
		goto exit;
	}
#endif

	/* configuration parameters */
	cq_read_count = ofi_nccl_cq_read_count();

	topo = nccl_ofi_topo_create();
	if (!topo) {
		NCCL_OFI_WARN("Failed to create NCCL OFI topology");
		ret = -ENODEV;
		goto exit;
	}

	PlatformManager::register_all_platforms(topo);

	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Plugin selected platform: %s",
	       PlatformManager::get_global().get_platform().get_name());

	ret = PlatformManager::get_global().get_platform().init(&provider_filter);
	if (ret != 0)
		goto exit;

	if (ofi_nccl_progress_model.get_source() != ParamSource::DEFAULT) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Requesting progress model %s",
			      ofi_nccl_progress_model.get_string());
	}

	/* This is ugly, but here's the basic protocol selection
	 * logic:
	 *   1. if the user set OFI_NCCL_PROTOCOL, use that.
	 *   2. if the platform init set nccl_ofi_selected_protocol,
	 *      use that.
	 *   3. If the rdma protocol reports multiple nics per device
	 *      and initialized successfully, use that.
	 *   4. If the sendrecv protocol initialized successfully, use
	 *      that
	 *   5. If the rdma protocol initialized successfully, use
	 *      that.
	 */
	if (ofi_nccl_protocol.get_source() == ParamSource::ENVIRONMENT) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using transport protocol %s (user set)",
			      ofi_nccl_protocol.get_string());
	} else if (ofi_nccl_protocol.get_source() == ParamSource::API) {
		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using transport protocol %s (platform set)",
			      ofi_nccl_protocol.get_string());
	}

	if (ofi_nccl_protocol.get_source() != ParamSource::DEFAULT) {
		bool dummy;

		if (ofi_nccl_protocol.get() == PROTOCOL::SENDRECV) {
			ret = nccl_net_ofi_sendrecv_init(provider_filter, &plugin);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to initialize sendrecv protocol");
				goto exit;
			}
		} else if (ofi_nccl_protocol.get() == PROTOCOL::RDMA) {
			ret = nccl_net_ofi_rdma_init(provider_filter, &plugin, &dummy, topo);
			if (ret != 0) {
				NCCL_OFI_WARN("Failed to initialize rdma protocol");
				goto exit;
			}
		}
	} else {
		bool have_multiple_rails = false;
		nccl_net_ofi_plugin_t *rdma_plugin = NULL, *sendrecv_plugin = NULL;

		try {
			ret = nccl_net_ofi_rdma_init(provider_filter, &rdma_plugin,
						     &have_multiple_rails, topo);
		}
		catch (const std::exception &e) {
			NCCL_OFI_WARN("Caught exception in rdma_init: %s", e.what());
			ret = -EINVAL;
		}
		if (ret != 0) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Failed to initialize rdma protocol: %s", fi_strerror(-ret));
			have_multiple_rails = false;
			rdma_plugin = NULL;
		}

		if (!have_multiple_rails || rdma_plugin == NULL) {
			try {
				ret = nccl_net_ofi_sendrecv_init(provider_filter,
								 &sendrecv_plugin);
			}
			catch (const std::exception &e) {
				NCCL_OFI_WARN("Caught exception in sendrecv_init: %s", e.what());
				ret = -EINVAL;
			}
			if (ret != 0) {
				NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
					       "Failed to initialized sendrecv protocol: %s", fi_strerror(-ret));
				sendrecv_plugin = NULL;
			}
		}

		if (have_multiple_rails && rdma_plugin != NULL) {
			ofi_nccl_protocol.set(PROTOCOL::RDMA);
			plugin = rdma_plugin;
			if (sendrecv_plugin != NULL) {
				delete sendrecv_plugin;
			}
		} else {
			ofi_nccl_protocol.set(PROTOCOL::SENDRECV);
			plugin = sendrecv_plugin;
			if (rdma_plugin != NULL) {
				delete rdma_plugin;
			}
		}

		if (plugin == NULL) {
			NCCL_OFI_WARN("Unable to find a protocol that worked.  Failing initialization.");
			ret = -EINVAL;
			goto exit;
		}

		NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Using transport protocol %s",
			      ofi_nccl_protocol.get_string());
	}

	ret = plugin->complete_init();
	if (ret != 0) {
		NCCL_OFI_WARN("Failed to initialize %s protocol", ofi_nccl_protocol.get_string());
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
	device = plugin->get_device(0);

	/* get the endpoint from the default domain, domain_key = 0 */
	ep = device->get_ep(0);
	if (ep == nullptr) {
		goto exit;
	}
	ret = device->get_properties(&properties);
	if (ret != 0) {
		goto exit;
	}
	NCCL_OFI_INFO(NCCL_NET | NCCL_INIT, "Support for global registrations: %s",
		      (properties.regIsGlobal == 0) ? "false" : "true");
	NCCL_OFI_INFO(NCCL_NET | NCCL_INIT, "Support for DMA-BUF registrations: %s",
		      (properties.dmabuf_support == 0) ? "false" : "true");
	ret = ep->release_ep(false, false);
	if (ret != 0) {
		goto exit;
	}

	assert(support_gdr != GDR_UNKNOWN);

	/* we don't actually know if GDR is supported until we've
	 * created the first endpoint, so this check needs to be way
	 * down here
	 */
	if (ofi_nccl_nic_dup_conns.get() > 0 && support_gdr != GDR_UNSUPPORTED) {
		NCCL_OFI_WARN("NCCL_OFI_NIC_DUP_CONNS set on platform that supports GPUDirect RDMA.  This configuration is not supported.");
		ret = -ENOTSUP;
		goto exit;
	}
	/* Force SIMPLE protocol when using a provider that does not support
	 * GDR. NCCL disables the LL128 protocol in this case, but leaves the
	 * LL protocol enabled. Without GDR, the LL protocol polls on host
	 * memory for completion flags. In addition to being slow, this assumes
	 * that host memory is updated in 8 byte segments. However, most
	 * providers that do not support HMEM (like the tcp or sockets
	 * providers) do not make any guarantees about data delivery ordering.
	 * There is not a good way to ask Libfabric providers about their data
	 * delivery support in the general case, so take a conservative
	 * approach and force the simple protocol whenever using a provider
	 * that does not support HMEM.
	 */
	if (support_gdr != GDR_SUPPORTED) {
		ret = nccl_net_ofi_configure_nccl_proto_simple("GDR");
		if (ret != 0) {
			goto exit;
		}
	}

	env_manager::getInstance().update_environment(&environ);

	*plugin_p = plugin;

 exit:
	if (ret != 0) {
		if (topo) {
			nccl_ofi_topo_free(topo);
		}
		NCCL_OFI_WARN(PACKAGE_NAME " initialization failed");
	}
	// On success, topology ownership is transferred to plugin
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

	props->latency = ofi_nccl_net_latency.get();

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

	props->max_p2p_bytes = nic_prov->ep_attr->max_msg_size;
	props->max_coll_bytes = nic_prov->ep_attr->max_msg_size;

	/* Should be successful for ptrSupport invocation */
	return 0;
}


int nccl_net_ofi_plugin_t::nccl_net_ofi_info_properties(struct fi_info *nic_prov,
							int dev_id,
							int num_devices,
							nccl_ofi_properties_t *props)
{
	int ret = 0;
	struct fid_nic *nic_info = nullptr;
	const char *platform_type = nullptr;
	nccl_net_ofi_device_t *device = nullptr;

	memset(props, 0, sizeof(*props));

	device = this->get_device(dev_id);
	if (OFI_UNLIKELY(device == nullptr)) {
		NCCL_OFI_WARN("Error accessing device %i.", dev_id);
		ret = -ENOTSUP;
		goto error;
	}
	props->guid = device->guid;

	ret = set_nic_props_default(dev_id, nic_prov, props);
	if (ret != 0) {
		goto error;
	}

	/*
	 * Determine the scope of MRs for providers to report global registration
	 * support to NCCL.
	 * NCCL uses regIsGlobal to determine support for User Registrations via
	 * the NCCL API. If providers tie MRs to endpoints, the plugin can not
	 * support this model (since NCCL maintains a per-domain registration
	 * cache which requires (domain-)global registrations.
	 */
	if (nic_prov->domain_attr->mr_mode & FI_MR_ENDPOINT) {
		props->regIsGlobal = 0;
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Global registrations are not supported");
	} else {
		props->regIsGlobal = 1;
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Global registrations supported");
	}

	props->max_mr_key_size = nic_prov->domain_attr->mr_key_size;

	/* We only claim DMA-BUF is supported when GDR is supported. In NCCL's logic, when plugin
	 * claims support for `NCCL_PTR_DMABUF`, NCCL thinks that it should be able to use GDR.
	 * So, on platform that doesn't support GDR, we should not advertise support for DMA-BUF
	 * to prevent NCCL from using GDR.
	 */
	props->dmabuf_support = nccl_ofi_dmabuf_viable_and_supported(nic_prov) & (support_gdr == GDR_SUPPORTED);
	if (props->dmabuf_support) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "DMA-BUF support is advertised in properties.");
	}

	/*
	 * the rest of the checks require NIC attributes to be filled in by libfabric
	 */
	nic_info = nic_prov->nic;
	if (nic_info == nullptr) {
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
		assert(props->name != nullptr);
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
	NCCL_OFI_INFO(NCCL_INIT | NCCL_NET, "Selected provider is %s, fabric is %s (found %d nics)",
		      selected_provider->fabric_attr->prov_name,
		      selected_provider->fabric_attr->name,
		      num_providers);

	if (strncmp("efa", selected_provider->fabric_attr->prov_name, strlen("efa")) == 0) {
		if (FI_VERSION_LT(fi_version(), FI_VERSION(1, 22))) {
			NCCL_OFI_WARN("EFA provider requires at least libfabric version 1.22.0.");
			return -ENOTSUP;
		}
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

	/* Check provider's data progress model */
	if (selected_provider->domain_attr->data_progress == FI_PROGRESS_AUTO) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s uses FI_PROGRESS_AUTO data progress model",
					selected_provider->fabric_attr->prov_name);
		data_progress_auto = true;
	} else if (selected_provider->domain_attr->data_progress == FI_PROGRESS_MANUAL) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s uses FI_PROGRESS_MANUAL data progress model",
					selected_provider->fabric_attr->prov_name);
		data_progress_auto = false;
	} else {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Provider %s uses data progress model: %d",
					selected_provider->fabric_attr->prov_name,
					selected_provider->domain_attr->data_progress);
		data_progress_auto = false;
	}

	return 0;
}


nccl_net_ofi_plugin_t::~nccl_net_ofi_plugin_t()
{
	for (size_t i = 0 ; i < this->get_num_devices() ; i++) {
		if (this->p_devs[i] != nullptr) {
			this->p_devs[i]->release_device();
		}
	}
}


void nccl_net_ofi_device_t::remove_domain_from_map(nccl_net_ofi_domain_t *domain)
{
	size_t n_removed = 0;

	assert(!this->domain_table.empty());
	for (auto it = this->domain_table.begin(); it != this->domain_table.end();) {
		if (it->second == domain) {
			it = this->domain_table.erase(it);
			++n_removed;
		} else {
			++it;
		}
	}

	assert_always(n_removed == 1);
}


nccl_net_ofi_domain_t *nccl_net_ofi_device_t::nccl_net_ofi_device_get_domain_impl(unsigned int domain_key)
{
	nccl_net_ofi_domain_t *domain = nullptr;

	assert(this->plugin != nullptr);

	auto domain_iter = this->domain_table.find(domain_key);

	if (domain_iter != this->domain_table.end()) {
		domain = domain_iter->second;
	} else {
		domain = this->create_domain();
		if (domain == nullptr) {
			NCCL_OFI_WARN("Initializing a new domain for device %s failed",
				      this->name);
			return nullptr;
		}

		this->domain_table.insert(std::pair(domain_key, domain));

		NCCL_OFI_TRACE(NCCL_NET, "Domain %p for device #%d (%s) is created",
			       domain,
			       this->dev_id,
			       this->name);
	}

	return domain;
}


nccl_net_ofi_domain_t *nccl_net_ofi_device_t::get_domain(unsigned int domain_key)
{
	nccl_net_ofi_domain_t *domain = nullptr;

	pthread_wrapper scoped_device_lock(&this->device_lock);
	domain = this->nccl_net_ofi_device_get_domain_impl(domain_key);

	return domain;
}


nccl_net_ofi_ep_t *nccl_net_ofi_device_t::get_ep(unsigned int domain_key)
{
	nccl_net_ofi_domain_t *domain = nullptr;
	nccl_net_ofi_ep_t *ep = nullptr;

	pthread_wrapper scoped_device_lock(&this->device_lock);

	domain = this->nccl_net_ofi_device_get_domain_impl(domain_key);
	if (domain == nullptr) {
		return nullptr;
	}

	ep = domain->get_ep();
	if (ep == nullptr) {
		return nullptr;
	}

	return ep;
}


nccl_net_ofi_device_t::nccl_net_ofi_device_t(nccl_net_ofi_plugin_t *plugin_arg,
					     int device_index,
					     struct fi_info *info)
	: plugin(plugin_arg),
	  dev_id(device_index),
	  name(strdup(info->fabric_attr->prov_name))
{
	int ret = 0;

	assert(this->plugin != nullptr);

	if (this->name == nullptr) {
		NCCL_OFI_WARN("Unable to allocate device name");
		throw std::runtime_error("Base device constructor: device name alloc failed");
	}

	PlatformManager::get_global().get_platform().device_set_guid(info, this);

	/* Intiaialize mutex for endpoint access */
	ret = nccl_net_ofi_mutex_init(&this->device_lock, nullptr);
	if (ret != 0) {
		NCCL_OFI_WARN("Unable to initialize device mutex");
		throw std::runtime_error("Base device constructor: device mutex init failed");
	}

	/* Initialize mr rkey handling */
	this->need_mr_rkey_pool = true;
	ret = nccl_ofi_mr_keys_need_own_key(info, &this->need_mr_rkey_pool);
	if (ret != 0) {
		NCCL_OFI_WARN("MR key config parsing failed: %s",
			      strerror(-ret));
		throw std::runtime_error("Base device constructor: MR key config parse failed");
	}
}


nccl_net_ofi_device_t::~nccl_net_ofi_device_t()
{
	if (this->name != nullptr) {
		free(this->name);
	}
}


int nccl_net_ofi_device_t::release_all_domain_and_ep()
{
	int ret, first_error = 0;

	nccl_net_ofi_ep_t *ep;

	pthread_wrapper scoped_device_lock(&this->device_lock);

	assert(!this->domain_table.empty());
	for (auto domain_iter = this->domain_table.begin() ;
	     domain_iter != this->domain_table.end();) {
		nccl_net_ofi_domain_t *domain = domain_iter->second;
		/* For each domain, clean up its endpoints. */
		nccl_net_ofi_mutex_lock(&domain->domain_lock);
		if (domain->get_endpoint_ptr()) {
			ep = domain->get_endpoint_ptr();
			domain->clear_endpoint();

			ret = ep->release_ep(true, true);
			if (ret != 0) {
				NCCL_OFI_WARN("Freeing endpoint failed: %d", ret);
				if (first_error != 0) {
					first_error = ret;
				}
			}
			ep = nullptr;
		}
		nccl_net_ofi_mutex_unlock(&domain->domain_lock);

		/* The call to domain->release() below will remove this domain
		   from the table, invalidating domain_iter. So increment it
		   here first. */
		++domain_iter;

		/* domain->release takes the domain lock, and removes itself
		 * from domain_table. Skipping device lock here.*/
		ret = domain->release_domain(true, true);
		if (ret != 0 && first_error != 0) {
			first_error = ret;
		}

	}

	if (OFI_UNLIKELY(!this->domain_table.empty())) {
		NCCL_OFI_WARN("%zu domains still active after cleanup",
			      this->domain_table.size());
		if (first_error != 0) {
			first_error = -FI_EBUSY; // Anything else than above
		}
	}

	return first_error;
}


void nccl_net_ofi_domain_t::remove_ep_from_map(nccl_net_ofi_ep_t *ep)
{
	size_t n_removed = 0;

	assert(!this->ep_table.empty());
	for (auto it = this->ep_table.begin(); it != this->ep_table.end();) {
		if (it->second == ep) {
			it = this->ep_table.erase(it);
			++n_removed;
		} else {
			++it;
		}
	}

	assert_always(n_removed == 1);
}


nccl_net_ofi_ep_t *nccl_net_ofi_domain_t::get_ep()
{
	nccl_net_ofi_ep_t *ep = nullptr;

	pthread_wrapper scoped_domain_lock(&this->domain_lock);

	long lookup_key = nccl_net_ofi_gettid();

	auto ep_iter = this->ep_table.find(lookup_key);

	if (ep_iter != this->ep_table.end()) {
		ep = ep_iter->second;
	} else {
		ep = this->create_endpoint();
		if (ep == nullptr) {
			NCCL_OFI_WARN("Creating new endpoint for domain %p failed",
				      this);
			return nullptr;
		}

		this->ep_table.insert(std::pair(lookup_key, ep));
		this->increment_ref_cnt();

		NCCL_OFI_TRACE(NCCL_NET, "Endpoint %p for domain %p is created",
			       ep, this);
	}

	ep->increment_ref_cnt();
	return ep;
}


int nccl_net_ofi_domain_t::release_domain(bool skip_device_lock, bool force_cleanup)
{
	int ret = 0;
	nccl_net_ofi_device_t *device_ptr = this->device;

	nccl_net_ofi_mutex_lock(&this->domain_lock);

	this->decrement_ref_cnt();

	if (this->ref_cnt == 0 || force_cleanup) {

		/* If domain ref_cnt is 0, then there should be no remaining
		   endpoints */
		assert(this->ref_cnt != 0 || this->endpoint == nullptr);

		// The caller takes device_lock when force_cleanup.
		if (!skip_device_lock) {
			nccl_net_ofi_mutex_lock(&device_ptr->device_lock);
		}

		/* Remove this domain from the domain table */
		device_ptr->remove_domain_from_map(this);

		// domain->free below is going to free the domain lock
		// and we've removed the domain from the hash table,
		// so no one should have a reference to the domain at
		// this point and we can release the mutex.
		nccl_net_ofi_mutex_unlock(&this->domain_lock);

		ret = this->cleanup_resources();
		delete this;

		if (!skip_device_lock) {
			nccl_net_ofi_mutex_unlock(&device_ptr->device_lock);
		}
		if (ret != 0) {
			NCCL_OFI_WARN("Freeing domain failed: %d", ret);
			return ret;
		}
	} else {
		nccl_net_ofi_mutex_unlock(&this->domain_lock);
	}

	return 0;
}


nccl_net_ofi_domain_t::nccl_net_ofi_domain_t(nccl_net_ofi_device_t *device_arg)
	: device(device_arg),
	ref_cnt(0)
{
	int ret;

	assert(this->device != nullptr);

	ret = nccl_net_ofi_mutex_init(&this->domain_lock, nullptr);
	if (ret != 0) {
		NCCL_OFI_WARN("Unable to initialize domain mutex");
		throw std::runtime_error("base domain constructor: mutex init failed");
	}

	if (!ofi_nccl_mr_cache_disable()) {
		this->mr_cache =
			nccl_ofi_mr_cache_init(NCCL_OFI_MR_CACHE_INIT_SIZE,
					       system_page_size);
		if (!this->mr_cache) {
			NCCL_OFI_WARN("Unable to initialize domain mr cache");
			throw std::runtime_error("base domain constructor: mr cache init failed");
		}
	}

	if (this->device->need_mr_rkey_pool) {
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
			throw std::runtime_error("base domain constructor: invalid size for mr keypool");
		}
		this->mr_rkey_pool = new nccl_ofi_idpool_t(1 << shift);
	} else {
		/* Mark key pool as not in use */
		this->mr_rkey_pool = new nccl_ofi_idpool_t(0);
	}
}


int nccl_net_ofi_domain_t::release_all_ep()
{
	int ret, first_error = 0;

	pthread_wrapper scoped_domain_lock(&this->domain_lock);

	assert(!this->ep_table.empty());
	for (auto ep_iter = this->ep_table.begin() ;
	     ep_iter != this->ep_table.end();) {
		nccl_net_ofi_ep_t *ep = ep_iter->second;

		/* The call to ep->release() below will remove this ep
		 * from the table, invalidating ep_iter. So increment it
		 * here first.
		 */
		++ep_iter;

		/* ep->release takes the domain lock, and removes itself
		 * from ep_table. Skipping device lock here.
		 */
		ret = ep->release_ep(true, true);
		if (ret != 0 && first_error != 0) {
			first_error = ret;
		}
	}

	if (this->unreleased_inactive_ep_counter > 0) {
		NCCL_OFI_WARN("%zu inactive endpoint are still open after cleanup",
			       this->unreleased_inactive_ep_counter);

		if (first_error != 0) {
			first_error = -FI_EBUSY; // Anything else than above
		}
	}

	if (OFI_UNLIKELY(!this->ep_table.empty())) {
		NCCL_OFI_WARN("%zu endpoint still active after cleanup",
			       this->ep_table.size());
		if (first_error != 0) {
			first_error = -FI_EBUSY; // Anything else than above
		}
	}
	return first_error;
}


nccl_net_ofi_domain_t::~nccl_net_ofi_domain_t()
{
	if (mr_cache != nullptr) {
		nccl_ofi_mr_cache_finalize(mr_cache);
	}

	if (mr_rkey_pool != nullptr) {
		delete mr_rkey_pool;
		mr_rkey_pool = nullptr;
	}
}


void nccl_net_ofi_ep_t::invalidate()
{
	if (this->ep_active == true) {
		this->ep_active = false;

		pthread_wrapper lock(&this->domain->domain_lock);

		/* Remove this endpoint from the thread->domain table so that it
		   is not used for future communicators */
		this->domain->remove_ep_from_map(this);

		this->domain->inc_unreleased_inactive_ep_counter();
	}
}


int nccl_net_ofi_ep_t::release_ep(bool skip_lock, bool force_cleanup)
{
	int ret = 0;
	nccl_net_ofi_domain_t *domain_ptr = this->domain;

	if (!skip_lock) {
		nccl_net_ofi_mutex_lock(&domain_ptr->domain_lock);
	}

	this->decrement_ref_cnt();

	/* Store ref_cnt in local variable in case the endpoint gets deleted */
	int local_ref_cnt = this->ref_cnt;
	
	if (local_ref_cnt == 0 || force_cleanup) {
		/* If this was the endpoint we stored in domain for connection
		   management, remove that reference as well */
		if (domain_ptr->get_endpoint_ptr() == this) {
			domain_ptr->clear_endpoint();
		}

		if (force_cleanup && local_ref_cnt != 0) {
			NCCL_OFI_INFO(NCCL_NET, "Endpoint %p still have ref count %d when released",
				      this, local_ref_cnt);
		}

		/* Remove this ep from the ep table if it was active.
		   If not, it was already removed in ep_invalidate. */
		if (this->ep_active) {
			domain_ptr->remove_ep_from_map(this);
		} else {
			domain_ptr->dec_unreleased_inactive_ep_counter();
		}

		ret = this->cleanup_resources();
		delete this;
	}

	if (!skip_lock) {
		nccl_net_ofi_mutex_unlock(&domain_ptr->domain_lock);
	}

	/* If we freed the endpoint (local_ref_cnt == 0), also release the domain
	 * (decrement its ref_cnt)
	 *
	 * Skip domain->release when handled by device->release_all_domain_and_ep()
	 * to avoid domain lock issue after the domain freed */
	if (!force_cleanup && ret == 0 && local_ref_cnt == 0) {
		ret = domain_ptr->release_domain(skip_lock, false);
	}

	return ret;
}


nccl_net_ofi_ep_t::nccl_net_ofi_ep_t(nccl_net_ofi_domain_t *domain_arg)
	: ep_active(true),
	  domain(domain_arg),
	  ref_cnt(0)
{
	assert(domain_arg != nullptr);

	/* Intiaialize mutex for endpoint access */
	int ret = nccl_net_ofi_mutex_init(&this->ep_lock, nullptr);
	if (ret != 0) {
		NCCL_OFI_WARN("Unable to initialize endpoint mutex");
		throw std::runtime_error("Base endpoint constructor: endpoint mutex init failed");
	}
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


int nccl_net_ofi_configure_nccl_proto_simple(const char *log_reason)
{
	NCCL_OFI_INFO(NCCL_INIT, "Need to force simple protocol: %s not supported", log_reason);
	env_manager::getInstance().insert_envvar("NCCL_PROTO", "simple", false);

	return 0;
}
