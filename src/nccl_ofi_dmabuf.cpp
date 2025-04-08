/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <stdio.h>
#include <sys/utsname.h>

#include "nccl_ofi_dmabuf.h"
#include "nccl_ofi_log.h"
#include "nccl_ofi_param.h"
#if HAVE_CUDA
#include "nccl_ofi_cuda.h"
#endif

/*
 * @brief call uname(2), parse major/minor kernel version and compare against
 * min version containing rdma dmabuf import ioctls.
 *
 * dmabuf rdma ioctls were added in 5.12: ref:
 *  github.com/torvalds/linux/commit/bfe0cc6
 */

static bool kernel_version_rdma_dmabuf_ioctl_ok(void)
{
	struct utsname buf = {};
	int maj = 0;
	int min = 0;
	errno = 0;
	if (uname(&buf) != 0) {
		errno = 0;
		return false;
	}

	if (sscanf(buf.release, "%d.%d", &maj, &min) != 2) {
		return false;
	}

	return (maj == 5 && min >= 12) || (maj > 5);
}

/**
 * @brief device_supports_dmabuf - Check if the device supports DMA-BUF
 * @info: Provider info or hints
 * @is_prov: true if info is provider, false if hints
 *
 * Determines if the device supports DMA-BUF based on its device ID
 * When passed hints (is_prov = false), it first
 * gets the actual provider info.
 *
 * Return: true if DMA-BUF is supported, false otherwise
 */
static bool device_supports_dmabuf(const struct fi_info *info, bool is_prov)
{
	struct fi_info *providers = NULL;
	char *endptr;
	uint32_t device_id;
	bool ret = false;

	if (!info) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Invalid input: info is NULL");
		goto error;
	}

	/* If hints provided, get actual provider info */
	if (!is_prov) {
		int rc = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, info, &providers);
		if (rc != 0 || !providers) {
			NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
				       "Failed to get provider info from hints");
			goto error;
		}
		info = providers;
	}

	/* Check for valid device attributes */
	if (!info->nic || !info->nic->device_attr || !info->nic->device_attr->device_id) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Invalid device attributes");
		goto error;
	}

	/* Convert device ID from string to uint32_t */
	device_id = (uint32_t)strtoul(info->nic->device_attr->device_id, &endptr, 16);
	if (*endptr != '\0') {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
			       "Invalid device ID format");
		goto error;
	}

	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Device ID: 0x%x", device_id);
	ret = (device_id >= 0xefa3);
	goto exit;

error:
	ret = false;

exit:
	/* Clean up provider info if we allocated it */
	if (providers)
		fi_freeinfo(providers);
	return ret;
}


/* Check preconditions for using DMA-BUF. Note that we may disable DMA-BUF for
 * other reasons, even if this function returns true. For example, if we do not
 * resolve a provider with FI_HMEM support */
int nccl_ofi_dmabuf_viable(const struct fi_info* info, bool is_prov)
{
	/* Disable DMA-BUF if building against older libfabric. */
	if (!HAVE_DECL_FI_MR_DMABUF) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Will not use DMA-BUF, requires Libfabric 1.20 or greater.");
		return false;
	}

	/* Disable DMA-BUF if explicitly disabled by user. */
	if (ofi_nccl_disable_dmabuf()) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Will not attempt to use DMA-BUF, explicitly disabled by user.");
		return false;
	}

	/* Disable DMA-BUF if using CUDA and CUDA does not report DMA-BUF
	 * support in device attributes. */
#if HAVE_CUDA
	if (!nccl_net_ofi_cuda_have_dma_buf_attr()) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
		               "Will not attempt to use DMA-BUF, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED was false.");
		return false;
	}
#endif

	/*
	 * Disable DMA-BUF if running on a kernel with version < 5.12
	 *
	 * Neither NCCL nor libfabric does this check. In the case of CUDA, the
	 * call above is not specific to RDMA -- ie: an H100 operating atop the
	 * nvidia-open and kernel 5.10 is capable of exporting dmabufs for
	 * consumption by mesa/v4l/etc, but the MRs will fail because the ioctls
	 * for importing those dmabufs don't yet exist.
	 */
	if (!kernel_version_rdma_dmabuf_ioctl_ok()) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Will not attempt to use DMA-BUF, kernel 5.12+ not found.");
		return false;
	}

	/* Check if device supports DMA-BUF */
	if (!device_supports_dmabuf(info, is_prov)) {
		NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "Will not attempt to use DMA-BUF, device ID does not support it.");
		return false;
	}

	/* When using an HMEM capable provider at API version 1.20 or greater,
	 * advertise DMA-BUF support in NCCL getProperties calls. When given dmabuf
	 * file descriptors from NCCL, forward them in fi_regattr calls and pass the
	 * FI_MR_DMABUF flag. */
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
	               "Attempting to use DMA-BUF capable providers. set OFI_NCCL_DISABLE_DMABUF=1 to disable");
	return true;
}
