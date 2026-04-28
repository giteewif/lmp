// ----------------------------------------------------------------------------
//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "concurrent_queue.h"
#include "concurrent_vector.h"
#include "memory_state.h"

struct Batch {
  size_t chunk_id_ = 0;
  size_t size_ = 0;
};
typedef ConcurrentVector<Batch> BatchVector;

struct GpuBatch {
  uint64_t task_id_ = 0;
  size_t chunk_id_ = 0;
  size_t chunk_offset_ = 0;
  size_t size_ = 0;
  size_t gpu_offset_ = 0;
  size_t handle_idx_ = 0;
};
typedef ConcurrentQueue<GpuBatch> BatchQueue;

struct FileChunk {
  int fd_;
  size_t file_offset_;
  size_t size_;
  size_t chunk_id_;
  size_t chunk_offset_;
};

#define KB (1024LL)
#define MB (1024LL * KB)
#define GB (1024LL * MB)

// using DeviceMap = std::unordered_map<std::string, int>;
struct MemCopyChunk {
  size_t src_offset_ = 0;
  size_t size_ = 0;
  size_t dst_offset_ = 0;
  size_t handle_idx_ = 0;
  uint64_t task_id_ = 0;
  uint8_t priority_ = 0;
  bool reorder_hint_ = false;
};
using MemCopyChunkList = std::vector<MemCopyChunk>;

struct MemCopyHandle {
  std::string cuda_ipc_handle_;
};
using MemCopyHandleList = std::vector<MemCopyHandle>;

typedef std::unordered_map<std::string, MemCopyHandleList> MemCopyHandleListMap;
typedef std::unordered_map<std::string, MemCopyChunkList> MemCopyChunkListMap;
typedef std::unordered_map<int, std::vector<void*>> MemPtrListMap;

// tensor_index
// INSERT_YOUR_CODE
// 结构类似于: [偏移量, 大小, shape, strides, dtype]
// 对应C++类型声明如下:
typedef std::tuple<
    uint64_t,                // offset
    uint64_t,                // size
    std::vector<size_t>,     // shape
    std::vector<size_t>,     // strides
    std::string              // dtype
> TensorIndexInfo;

typedef std::unordered_map<std::string, TensorIndexInfo> TensorIndexMap;
typedef std::unordered_map<std::string, TensorIndexInfo> TensorIndexResizeMap;

// device_id, chunk_offset, size, gpu_offset. handle_idx
typedef std::tuple<int, size_t, size_t, size_t, size_t> GpuChunk;

enum class CopyPriority : uint8_t { LOW = 0, HIGH = 1 };

inline uint64_t HashCombine64(uint64_t seed, uint64_t v) {
  seed ^= v + 0x9E3779B97F4A7C15ULL + (seed << 6) + (seed >> 2);
  return seed;
}

inline uint64_t BuildCopyTaskId(size_t src_offset, size_t size, size_t dst_offset,
                                int device_id, size_t handle_idx) {
  uint64_t seed = 0;
  seed = HashCombine64(seed, static_cast<uint64_t>(src_offset));
  seed = HashCombine64(seed, static_cast<uint64_t>(size));
  seed = HashCombine64(seed, static_cast<uint64_t>(dst_offset));
  seed = HashCombine64(seed, static_cast<uint64_t>(device_id));
  seed = HashCombine64(seed, static_cast<uint64_t>(handle_idx));
  return seed;
}

class LockFreeBitmap {
 public:
  explicit LockFreeBitmap(size_t num_bits)
      : num_bits_(num_bits), words_((num_bits + 63) / 64) {
    for (auto& word : words_) {
      word.store(0, std::memory_order_relaxed);
    }
  }

  bool test(size_t bit) const {
    const size_t idx = bit / 64;
    if (idx >= words_.size()) {
      return false;
    }
    const uint64_t mask = 1ULL << (bit % 64);
    return (words_[idx].load(std::memory_order_acquire) & mask) != 0;
  }

  bool test_and_set(size_t bit) {
    const size_t idx = bit / 64;
    if (idx >= words_.size()) {
      return false;
    }
    const uint64_t mask = 1ULL << (bit % 64);
    const uint64_t old =
        words_[idx].fetch_or(mask, std::memory_order_acq_rel);
    return (old & mask) != 0;
  }

  void clear(size_t bit) {
    const size_t idx = bit / 64;
    if (idx >= words_.size()) {
      return;
    }
    const uint64_t mask = ~(1ULL << (bit % 64));
    words_[idx].fetch_and(mask, std::memory_order_acq_rel);
  }

  void reset() {
    for (auto& word : words_) {
      word.store(0, std::memory_order_release);
    }
  }

  size_t num_bits() const { return num_bits_; }

 private:
  size_t num_bits_;
  std::vector<std::atomic<uint64_t>> words_;
};
