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

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Third-party library headers
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "error_handling.h"
#include "pinned_memory.h"
#include "types_and_defs.h"

struct GpuReplica {
  std::condition_variable cv_;
  MemoryState state_ = MemoryState::UNINITIALIZED;

  struct TaskPart {
    size_t chunk_id_ = 0;
    size_t chunk_offset_ = 0;
    size_t size_ = 0;
    size_t gpu_offset_ = 0;
    size_t handle_idx_ = 0;
  };

  struct CopyTask {
    uint64_t task_id_ = 0;
    int device_id_ = 0;
    std::atomic<uint8_t> priority_raw_{static_cast<uint8_t>(CopyPriority::LOW)};
    bool reorder_hint_ = false;
    std::vector<TaskPart> parts_;
    std::atomic<bool> ready_{false};
    std::atomic<int> exec_state_{0};  // 0: queued, 1: running, 2: finished

    CopyPriority priority() const {
      return static_cast<CopyPriority>(
          priority_raw_.load(std::memory_order_acquire));
    }

    void set_priority(CopyPriority priority) {
      priority_raw_.store(static_cast<uint8_t>(priority),
                          std::memory_order_release);
    }
  };

  struct PriorityGpuQueue {
    std::mutex mu_;
    std::condition_variable cv_;
    std::queue<uint64_t> high_queue_;
    std::queue<uint64_t> low_queue_;
    bool closed_ = false;

    void Push(uint64_t task_id, CopyPriority priority);
    bool Pop(uint64_t* task_id);
    void Close();
    size_t HighSize();
    size_t LowSize();
  };

  // 64M bits ~= 8MB per bitmap.
  static constexpr size_t kBitmapBits = (1ULL << 26);
  std::unordered_map<int, std::shared_ptr<PriorityGpuQueue>> gpu_loading_queue_;
  MemPtrListMap device_ptrs_;
  std::unordered_map<uint64_t, std::shared_ptr<CopyTask>> task_map_;
  LockFreeBitmap reorder_bitmap_{kBitmapBits};
  mutable std::shared_mutex task_mu_;
  std::mutex wait_mu_;
  std::condition_variable task_cv_;
  LockFreeBitmap completed_bitmap_{kBitmapBits};
  LockFreeBitmap enqueued_bitmap_{kBitmapBits};
  LockFreeBitmap enqueued_high_bitmap_{kBitmapBits};

  size_t TaskBit(uint64_t task_id) const {
    return task_id % completed_bitmap_.num_bits();
  }
};
using GpuReplicaPtr = std::shared_ptr<GpuReplica>;

class Model {
 public:
  Model(const std::filesystem::path& model_path) : model_path_(model_path) {}
  int Initialize(const std::filesystem::path storage_path);
  int AllocatePinnedMemory(std::shared_ptr<PinnedMemoryPool> pool);
  int AllocatePinnedMemory(std::shared_ptr<PinnedMemoryPoolShared> pool);
  std::vector<std::string> GetSharedMemoryNames();
  
  int ToHost(int num_threads);
  int ToHostResize(int num_threads);
  int EnsureGpuReplica(const std::string& replica_uuid);
  int ToGpu(const std::string& replica_uuid, const MemPtrListMap& device_ptrs,
            const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
            const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles);
  int WaitInHost();
  int WaitInGpu(const std::string& replica_uuid);
  int SubmitHighPriorityTasks(const std::string& replica_uuid,
                              const std::vector<uint64_t>& task_ids,
                              std::vector<uint64_t>* pending_task_ids);
  int SetReorderBitmap(const std::string& replica_uuid,
                       const std::vector<uint64_t>& task_ids);
  int WaitCopyTasks(const std::string& replica_uuid,
                    const std::vector<uint64_t>& task_ids, uint64_t timeout_ms,
                    std::vector<uint64_t>* pending_task_ids);
  int FreeGpu(const std::string& replica_uuid);
  int FreeHost();
  int TryFreeHost();
  uint64_t GetModelSize() const { return model_size_; }
  bool HasTensorIndexResize() const { return !tensor_index_resize_.empty(); }
  void SetTensorInfo(const TensorIndexMap& tensor_index, const TensorIndexResizeMap& tensor_index_resize);

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  MemoryState state_ = MemoryState::UNINITIALIZED;

  // Model path
  const std::string model_path_;

  // Model info needs to be initialized
  size_t model_size_;
  size_t model_size_resize_;
  TensorIndexMap tensor_index_;
  TensorIndexResizeMap tensor_index_resize_;

  std::vector<size_t> partition_sizes_;
  std::vector<std::filesystem::path> partition_paths_;
  std::shared_ptr<PinnedMemory> pinned_mem_;

  std::unordered_map<std::string, GpuReplicaPtr> gpu_replicas_;

  std::shared_ptr<BatchVector> host_ptr_vector_;

  std::vector<std::tuple<int, size_t, size_t>> MapDataToChunks(
      size_t offset, size_t size, size_t chunk_size);
  int DispatchToGpu(
      const std::shared_ptr<GpuReplica>& gpu_replica,
      const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
      const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles);
  int CopyTaskToGpu(const std::shared_ptr<GpuReplica>& gpu_replica,
                    const std::shared_ptr<GpuReplica::CopyTask>& task,
                    const std::vector<void*>& device_ptr_list,
                    const std::vector<char*>& host_buffers);
};
using ModelPtr = std::shared_ptr<Model>;