/*
 * Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#include "config.h"

#include "nccl_ofi_kvstore.h"

#include <cstdint>
#include <unordered_map>

using map_t = std::unordered_map<uint64_t, void*>;

class kvstore {
  map_t m{};

 public:
  std::size_t count() const { return m.size(); };

  int insert(uint64_t key, void* val) {
    if (m.count(key) != 0) {
      return -1;
    }

    m.emplace(key, val);
    return 0;
  };

  void* remove(uint64_t key) {
    if (auto it = m.find(key); it != m.end()) {
      auto const val = std::get<1>(*it);
      m.erase(it);
      return val;
    }
    return nullptr;
  };

  void* find(uint64_t key) const {
    if (auto it = m.find(key); it == m.end()) {
      return nullptr;
    } else {
      return std::get<1>(*it);
    }
  };
};

nccl_ofi_kvstore_t* nccl_ofi_kvstore_init() {
  return new kvstore();
};
void nccl_ofi_kvstore_fini(nccl_ofi_kvstore_t* ptr) {
  delete static_cast<kvstore*>(ptr);
};
std::size_t nccl_ofi_kvstore_count(nccl_ofi_kvstore_t* ptr) {
  return static_cast<kvstore*>(ptr)->count();
}
int nccl_ofi_kvstore_insert(
    nccl_ofi_kvstore_t* ptr, uint64_t key, void* value) {
  return static_cast<kvstore*>(ptr)->insert(key, value);
};
void* nccl_ofi_kvstore_remove(nccl_ofi_kvstore_t* ptr, uint64_t key) {
  return static_cast<kvstore*>(ptr)->remove(key);
};
void* nccl_ofi_kvstore_find(nccl_ofi_kvstore_t* ptr, uint64_t key) {
  return static_cast<kvstore*>(ptr)->find(key);
}
