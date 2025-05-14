/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions
 are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the name of NVIDIA CORPORATION, Lawrence Berkeley National
    Laboratory, the U.S. Department of Energy, nor the names of their
    contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The U.S. Department of Energy funded the development of this software
 under subcontract 7078610 with Lawrence Berkeley National Laboratory.
 *
 ************************************************************************/

/*
 * Note: Code in this file was taken directly from NCCL to guarantee that
 * getHostHash() returns the same value in our generated topology as the NCCL
 * generated topology, because NCCL does not properly add the host_hash field to
 * a minimal topology file.  THe path to removing this code is to remove the
 * need to generate topology files at all on NCCL versions that support
 * multinode NVL, which thankfully are also the versions that support the vNIC
 * interface.
 */

#include "config.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdlib.h>
#include <unistd.h>

#include "nccl_ofi.h"

// The hash isn't just a function of the bytes but also where the bytes are split
// into different calls to eatHash().
static inline void eatHash(uint64_t acc[2], const void* bytes, size_t size) {
  char const* ptr = (char const*)bytes;
  acc[0] ^= size;
  while (size != 0) {
    // Mix the accumulator bits.
    acc[0] += acc[1];
    acc[1] ^= acc[0];
    acc[0] ^= acc[0] >> 31;
    acc[0] *= 0x9de62bbc8cef3ce3;
    acc[1] ^= acc[1] >> 32;
    acc[1] *= 0x485cd6311b599e79;
    // Read in a chunk of input.
    size_t chunkSize = size < sizeof(uint64_t) ? size : sizeof(uint64_t);
    uint64_t x = 0;
    memcpy(&x, ptr, chunkSize);
    ptr += chunkSize;
    size -= chunkSize;
    // Add to accumulator.
    acc[0] += x;
  }
}

static inline uint64_t digestHash(uint64_t const acc[2]) {
  uint64_t h = acc[0];
  h ^= h >> 31;
  h *= 0xbac3bd562846de6b;
  h += acc[1];
  h ^= h >> 32;
  h *= 0x995a187a14e7b445;
  return h;
}

static inline uint64_t getHash(const void* bytes, size_t size) {
  uint64_t acc[2] = {1, 1};
  eatHash(acc, bytes, size);
  return digestHash(acc);
}

static int getHostName(char* hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return -1;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen-1)) i++;
  hostname[i] = '\0';
  return 0;
}

static uint64_t hostHashValue = 0;
/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of;
 *
 * $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 *
 * This string can be overridden by using the NCCL_HOSTID env var.
 */
#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"
static void getHostHashOnce() {
  char hostHash[1024];
  const char *hostId;

  // Fall back is the full hostname if something fails
  (void) getHostName(hostHash, sizeof(hostHash), '\0');
  int offset = strlen(hostHash);

  if ((hostId = getenv("NCCL_HOSTID")) != NULL) {
    strncpy(hostHash, hostId, sizeof(hostHash)-1);
    hostHash[sizeof(hostHash)-1] = '\0';
  } else {
    FILE *file = fopen(HOSTID_FILE, "r");
    if (file != NULL) {
      char *p;
      if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
      }
      fclose(file);
    }
  }

  // Make sure the string is terminated
  hostHash[sizeof(hostHash)-1]='\0';

  hostHashValue = getHash(hostHash, strlen(hostHash));
}

uint64_t getHostHash(void) {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, getHostHashOnce);
  return hostHashValue;
}


