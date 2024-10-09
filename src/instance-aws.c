/*
 * Copyright (c) 2018-2024 Amazon.com, Inc. or its affiliates. All rights
 * reserved. Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 */

#include "config.h"
#include <stdio.h>
#include <stdlib.h>

#include "nccl_ofi.h"
#include "nccl_ofi_platform.h"
#include "nccl_ofi_pthread.h"

static struct ec2_platform_data platform_data_map[] = {
    {
        .name = "p4d.24xlarge",
        .topology = "p4d-24xl-topo.xml",
        .default_dup_conns = 0,
        .latency = 75.0,
        .gdr_required = true,
        .net_flush_required = true,
        .default_protocol = "SENDRECV",
        .domain_per_thread = 0,
    },
    {
        .name = "p4de.24xlarge",
        .topology = "p4de-24xl-topo.xml",
        .default_dup_conns = 0,
        .latency = 75.0,
        .gdr_required = true,
        .net_flush_required = true,
        .default_protocol = "SENDRECV",
        .domain_per_thread = 0,
    },
    {
        .name = "p3dn.24xlarge",
        .topology = NULL,
        .default_dup_conns = 4,
        .latency = 150.0,
        .gdr_required = false,
        .net_flush_required = true,
        .default_protocol = "SENDRECV",
        .domain_per_thread = 0,
    },
    {
        .name = "p5.48xlarge",
        .topology = NULL,
        .default_dup_conns = 0,
        .latency = 75.0,
        .gdr_required = true,
        .net_flush_required = false,
        .default_protocol = "RDMA",
        .domain_per_thread = 0,
    },
    {
        .name = "p5e.48xlarge",
        .topology = NULL,
        .default_dup_conns = 0,
        .latency = 75.0,
        .gdr_required = true,
        .net_flush_required = false,
        .default_protocol = "RDMA",
        .domain_per_thread = 0,
    },
    {
        .name = "g5.48xlarge",
        .topology = "g5.48xl-topo.xml",
        .default_dup_conns = 0,
        .latency = 75.0,
        .gdr_required = false,
        .net_flush_required = true,
        .default_protocol = "SENDRECV",
        .domain_per_thread = 0,
    },
    {
        .name = "trn1.32xlarge",
        .topology = NULL,
        .default_dup_conns = 0,
        .latency = 75.0,
        .gdr_required = true,
        .net_flush_required = true,
        .default_protocol = "SENDRECV",
        .domain_per_thread = 1,
    },
    {
        .name = "trn1n.32xlarge",
        .topology = NULL,
        .default_dup_conns = 0,
        .latency = 75.0,
        .gdr_required = true,
        .net_flush_required = true,
        .default_protocol = "SENDRECV",
        .domain_per_thread = 1,
    },
    {
        .name = "trn2n.48xlarge",
        .topology = NULL,
        .default_dup_conns = 0,
        .latency = 75.0,
        .gdr_required = true,
        .net_flush_required = true,
        .default_protocol = "RDMA",
        .domain_per_thread = 1,
    }};

const char *get_platform_type(void) {
  char file[] = "/sys/devices/virtual/dmi/id/product_name";
  FILE *fd = NULL;
  char ch;
  size_t len = 0;
  size_t platform_type_len = 64;
  static bool init = false;
  static char *platform_type = NULL;
  static pthread_mutex_t platform_mutex = PTHREAD_MUTEX_INITIALIZER;

  nccl_net_ofi_mutex_lock(&platform_mutex);

  if (init) {
    nccl_net_ofi_mutex_unlock(&platform_mutex);
    return platform_type;
  }

  init = true;

  fd = fopen(file, "r");
  if (fd == NULL) {
    NCCL_OFI_WARN("Error opening file: %s", file);
    goto error;
  }

  platform_type = (char *)malloc(sizeof(char) * platform_type_len);
  if (platform_type == NULL) {
    NCCL_OFI_WARN("Unable to allocate platform type");
    goto error;
  }

  /* Read first line of the file, reallocing the buffer as necessary */
  while ((feof(fd) == 0) && (ferror(fd) == 0) && ((ch = fgetc(fd)) != '\n')) {
    platform_type[len++] = ch;
    if (len >= platform_type_len) {
      char *new_platform_type =
          (char *)realloc(platform_type, len + platform_type_len);
      if (new_platform_type == NULL) {
        NCCL_OFI_WARN("Unable to (re)allocate platform type");
        goto error;
      }
      platform_type = new_platform_type;
    }
  }

  platform_type[len] = '\0';

  if (ferror(fd)) {
    NCCL_OFI_WARN("Error reading file: %s", file);
    goto error;
  }

  NCCL_OFI_TRACE(NCCL_INIT | NCCL_NET, "EC2 platform type is %s",
                 platform_type);

  goto exit;

error:
  if (platform_type) {
    free(platform_type);
    platform_type = NULL;
  }

exit:
  if (fd)
    fclose(fd);

  nccl_net_ofi_mutex_unlock(&platform_mutex);

  return platform_type;
}

struct ec2_platform_data *get_platform_data() {
  static bool init = false;
  static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  static struct ec2_platform_data *platform_data = NULL;
  const size_t platform_n =
      sizeof(platform_data_map) / sizeof(platform_data_map[0]);
  const char *platform_type = NULL;

  nccl_net_ofi_mutex_lock(&mutex);

  if (init) {
    nccl_net_ofi_mutex_unlock(&mutex);
    return platform_data;
  }
  init = true;

  platform_type = get_platform_type();
  if (platform_type == NULL) {
    nccl_net_ofi_mutex_unlock(&mutex);
    return NULL;
  }

  for (size_t idx = 0; idx < platform_n; idx++) {
    if (strcmp(platform_type, platform_data_map[idx].name) == 0)
      platform_data = &platform_data_map[idx];
  }

  nccl_net_ofi_mutex_unlock(&mutex);

  return platform_data;
}
