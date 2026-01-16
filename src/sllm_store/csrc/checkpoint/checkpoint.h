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

#include <torch/extension.h>
#include <torch/script.h>  // One-stop header.
#include "sllm_store/types_and_defs.h"

#include <string>
#include <unordered_map>

std::unordered_map<std::string, uint64_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path);

std::unordered_map<std::string, torch::Tensor> RestoreTensors2(
    const std::unordered_map<
        std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const std::unordered_map<int, void*>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        tensor_device_offsets);
    
std::unordered_map<std::string, torch::Tensor> RestoreTensors(
    const std::unordered_map<
        std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const std::unordered_map<int, void*>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        tensor_device_offsets);

// {dev_id: ptr}
std::unordered_map<int, void*> AllocateCudaMemory(
    const std::unordered_map<int, size_t>& tensor_sizes);
void FreeCudaMemory(
    const std::unordered_map<int, void*>& memory_ptr);
std::unordered_map<int, std::string> GetCudaMemoryHandles(
    const std::unordered_map<int, void*>& memory_ptrs);
std::unordered_map<int, std::vector<std::string>> GetCudaMemoryHandles(
    const std::unordered_map<int, std::vector<void*>>& memory_ptrs);
std::unordered_map<int, std::string> GetDeviceUuidMap();

std::unordered_map<std::string, int> GetGpuUUID();

// 从共享内存构造基于 server 端管理的 pinned memory 的 CPU tensor
// memory_base_address: {chunk_id: base_ptr} - 每个 chunk 的基地址（通过共享内存或 mmap 获得）
// tensor_metadata: tensor 的元数据，包含 offset, size, shape, strides, dtype
// chunk_size: pinned memory 的 chunk 大小
std::unordered_map<std::string, torch::Tensor> RestoreTensorsFromPinnedMemory(
    const std::unordered_map<int, void*>& memory_base_address,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size);

// 从共享内存名称列表创建 CPU tensor
// shm_names: 共享内存名称列表
// tensor_metadata: tensor 的元数据，包含 offset, size, shape, strides, dtype
// chunk_size: pinned memory 的 chunk 大小
std::unordered_map<std::string, torch::Tensor> RestoreTensorsFromSharedMemoryNames(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size);

// 将一个 layer 的若干个专家映射到连续虚拟空间中
// shm_names: 共享内存名称列表
// tensor_metadata: tensor 的元数据，包含 offset, size, shape, strides, dtype
// chunk_size: pinned memory 的 chunk 大小
// name_continuous_space: 需要映射到连续地址空间的 tensor 名称列表
std::unordered_map<std::string, torch::Tensor> RestoreExpertsFromSharedMemory(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::string>& name_continuous_space);

// 优化版本：预先打开所有 chunk 文件描述符，按 chunk 分组批量处理
// 主要优化：
// 1. 预先打开所有需要的 chunk 文件描述符，减少系统调用
// 2. 按 chunk 分组 tensor，批量处理，减少重复映射检查
// 3. 对同一 chunk 的 tensor 按偏移排序，提高缓存局部性
std::unordered_map<std::string, torch::Tensor> RestoreExpertsFromSharedMemoryOptimized(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::string>& name_continuous_space);

// 支持多个 vector，为每个 vector 创建一个 big_tensor
// shm_names: 共享内存名称列表
// tensor_metadata: tensor 的元数据
// chunk_size: pinned memory 的 chunk 大小
// name_groups: 多个 tensor name vectors，每个 vector 中的 tensor 会被拼成一个大 tensor
// 返回: map 中包含每个 group 的 big_tensor（key: "group_0_big_tensor", "group_1_big_tensor", ...）
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemory(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups);

// 性能分析版本：使用 std::cerr 打印各部分耗时，用于找出性能瓶颈
// 参数与 RestoreExpertsGroupsFromSharedMemory 相同
// 输出详细的性能分析信息到 stderr，包括：
// - 各步骤耗时及占比
// - shm_open、mmap、memcpy 调用次数和总数据量
// - 跨 chunk tensor 的处理时间
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemoryProfiled(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups);

// 静默版本：功能与 RestoreExpertsGroupsFromSharedMemoryProfiled 相同，但不输出任何日志
// 用于生产环境，避免性能分析输出影响性能
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemorySilent(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups);

// 支持多个 vector，为每个 vector 创建一个 big_tensor（带缓存版本）
// 这个版本会复用已映射的内存，避免频繁的 mmap/munmap 操作
// 参数与 RestoreExpertsGroupsFromSharedMemory 相同
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemoryCached(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups);

// 主动释放所有缓存的 group 内存映射
// 注意：只有当所有引用计数为 0 时才会真正释放内存
void ReleaseCachedGroupMemory();

// TensorIndexResizeMap 缓存包装类，用于避免重复转换 Python dict
// 在 Python 端只转换一次，然后重复使用
class TensorIndexResizeMapCache {
public:
  TensorIndexResizeMapCache(const TensorIndexResizeMap& metadata) 
    : metadata_(metadata) {}
  
  const TensorIndexResizeMap& get() const { return metadata_; }
  
private:
  TensorIndexResizeMap metadata_;
};

// 使用缓存的版本，避免每次调用都转换 tensor_metadata
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemoryProfiledCached(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMapCache& tensor_metadata_cache,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups);

// 静默版本的缓存版本，避免每次调用都转换 tensor_metadata
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemorySilentCached(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMapCache& tensor_metadata_cache,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups);