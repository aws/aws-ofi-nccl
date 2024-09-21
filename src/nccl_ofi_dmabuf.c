/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include <errno.h>
#include <stdbool.h>
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


/* Check preconditions for using DMA-BUF. Note that we may disable DMA-BUF for
 * other reasons, even if this function returns true. For example, if we do not
 * resolve a provider with FI_HMEM support */
int nccl_ofi_dmabuf_viable()
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

	/* When using an HMEM capable provider at API version 1.20 or greater,
	 * advertise DMA-BUF support in NCCL getProperties calls. When given dmabuf
	 * file descriptors from NCCL, forward them in fi_regattr calls and pass the
	 * FI_MR_DMABUF flag. */
	NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET,
	               "Attempting to use DMA-BUF capable providers. set OFI_NCCL_DISABLE_DMABUF=1 to disable");
	return true;
}
