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

// Include glog BEFORE model.h to ensure glog's LOG macro takes precedence
// over PyTorch's LOG macro (which is included via model.h -> torch headers)
#include <glog/logging.h>

#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Third-party library headers
#include <cuda_runtime.h>

// Own Headers
#include "cuda_memory.h"
// #include "cuda_memory_pool.h"
#include "model.h"
#include "pinned_memory.h"
#include "pinned_memory_pool.h"
#include "pinned_memory_pool_shared.h"
#include "types_and_defs.h"

class CheckpointStore {
 public:
  CheckpointStore(const std::string& storage_path, size_t memory_pool_size,
                  int num_thread, size_t chunk_size,
                  bool use_shared_memory = false,
                  const std::string& shm_name_prefix = "/sllm_pinned_pool");
  ~CheckpointStore();

  int64_t RegisterModelInfo(const std::string& model_path,
                            const TensorIndexMap& tensor_index,
                            const TensorIndexResizeMap& tensor_index_resize);
  int LoadModelFromDisk(const std::string& model_path);
  int LoadModelFromDiskAsync(const std::string& model_path);
  int LoadModelFromMem(const std::string& model_path,
                       const std::string& replica_uuid,
                       const MemCopyHandleListMap& gpu_memory_handles,
                       const MemCopyChunkListMap& mem_copy_chunks,
                       const bool& use_fixed_gpu_ptrs);
  int LoadModelFromMemAsync(const std::string& model_path,
                            const std::string& replica_uuid,
                            const MemCopyHandleListMap& gpu_memory_handles,
                            const MemCopyChunkListMap& mem_copy_chunks,
                            const bool& use_fixed_gpu_ptrs);
  int WaitModelInGpu(const std::string& model_path,
                     const std::string& replica_uuid);
  int UnloadModelFromHost(const std::string& model_path);
  int ClearMem();

  int RegisterFixedGpuPtrs(
    const MemCopyHandleListMap& memory_handles);
    int ReleaseRegisteredFixedGpuPtrsAll();
    int ReleaseRegisteredFixedGpuPtrs(
      const MemCopyHandleListMap& memory_handles);
  
  // Get shared memory names for a model
  // Returns a pair of (shared memory names, chunk_size)
  std::pair<std::vector<std::string>, size_t> GetModelSharedMemoryNames(const std::string& model_path);

 public:
  // Get methods
  size_t GetMemPoolSize() const { return memory_pool_size_; }
  size_t GetChunkSize() const { return chunk_size_; }
  bool IsUsingSharedMemory() const { return use_shared_memory_; }

 private:
  // A GPU info struct
  struct GpuInfo {
    // uuid
    std::string uuid_;
    // int device_id_;
    size_t total_memory_ = 0;
    size_t free_memory_ = 0;
    cudaStream_t stream_;
  };

  const std::filesystem::path storage_path_;
  int num_gpus_;
  std::unordered_map<int, GpuInfo> gpu_info_map_;
  std::unordered_map<std::string, std::shared_ptr<Model>> model_map_;
  std::unordered_map<std::string,
                     std::chrono::time_point<std::chrono::system_clock>>
      model_last_access_time_;
  std::mutex model_info_mutex_;
  const size_t memory_pool_size_;
  std::shared_ptr<PinnedMemoryPool> memory_pool_;
  std::shared_ptr<PinnedMemoryPoolShared> memory_pool_shared_;
  bool use_shared_memory_;
  std::string shm_name_prefix_;
  int num_thread_;
  size_t chunk_size_;

  std::queue<std::future<int>> async_tasks_;

  // add memory handles
  MemPtrListMap registered_fixed_gpu_ptrs_;
  // map string to ptr, string -> handle -> ptr
  DeviceStringMemPtrMap registered_device_string_mem_ptr_map_;

  size_t GetNumChunkFromTensorSize(size_t tensor_size);
  ModelPtr GetModelPtr(const std::string& model_path);
  GpuReplicaPtr NewGpuReplica(const std::shared_ptr<Model>& model,
                              const std::string& replica_uuid);
  int InitializeModel(const std::shared_ptr<Model>& model);
  int AllocatePinnedMemory(const std::shared_ptr<Model>& model);
  int AllocatePinnedMemoryShared(const std::shared_ptr<Model>& model);
  std::vector<std::tuple<int, size_t, size_t>> CalculateChunks(size_t offset,
                                                               size_t size);
  int AllocateCudaMemory(
      const std::shared_ptr<GpuReplica>& gpu_replica,
      std::vector<std::pair<int, uint64_t>> gpu_memory_sizes);
  ModelPtr GetModelByName(const std::string& model_path);
  MemPtrListMap GetDevicePtrsFromMemHandles(
      const MemCopyHandleListMap& memory_handles);
  
  MemPtrListMap GetFixedDevicePtrsFromMemHandles(
    const MemCopyHandleListMap& memory_handles);
};