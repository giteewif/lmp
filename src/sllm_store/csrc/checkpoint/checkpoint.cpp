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
#include "checkpoint.h"

#include <ATen/cuda/CUDABlas.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <errno.h>
#include <fcntl.h>
#include <nvml.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <unistd.h>
#include <cstring>

#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "progress_bar.h"
#include "tensor_writer.h"
#include "sllm_store/types_and_defs.h"
#include <regex>

#define BUFFER_SIZE 1 << 30

std::unordered_map<std::string, uint64_t> SaveTensors(
    std::vector<std::string> tensor_names,
    std::unordered_map<std::string, std::pair<uint64_t, uint64_t>>& tensor_data,
    const std::string& path) {
  std::string tensor_filename = std::filesystem::path(path) / "tensor.data";
  // make a tensor writer
  TensorWriter writer(tensor_filename);
  // make a tensor index
  std::unordered_map<std::string, uint64_t> tensor_offsets;
  // Some tensors may share the same data, so we need to record the data to
  // avoid duplication
  std::unordered_map<const char*, std::string> data_record;

  int total = tensor_names.size();
  int count = 0;

  for (const auto& name : tensor_names) {
    const auto& [base, size] = tensor_data[name];
    const char* data_ptr = reinterpret_cast<const char*>(base);
    if (data_record.find(data_ptr) != data_record.end()) {
      tensor_offsets[name] = tensor_offsets[data_record[data_ptr]];
      continue;
    }
    data_record[data_ptr] = name;

    std::cout << "help";
    uint64_t offset = writer.writeRecord(data_ptr, size);
    tensor_offsets[name] = offset;

    // Update progress bar
    count++;
    float progress = static_cast<float>(count) / total;
    showProgressBar(progress, "Saving tensors: ");
  }

  return tensor_offsets;
}

// Function to print the binary array in hexadecimal format
void printBinaryArrayInHex(const unsigned char* data, size_t size) {
  std::cout << "Data in Hex: ";
  for (size_t i = 0; i < size; ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
  }
  std::cout << std::dec
            << std::endl;  // Switch back to decimal for any future output
}

// Mapping from string to at::ScalarType
at::ScalarType stringToScalarType(const std::string& dtype_str) {
  static const std::unordered_map<std::string, at::ScalarType> dtype_map = {
      {"torch.float16", torch::kFloat16},  {"torch.float32", torch::kFloat32},
      {"torch.float64", torch::kFloat64},  {"torch.int16", torch::kInt16},
      {"torch.int32", torch::kInt32},      {"torch.int64", torch::kInt64},
      {"torch.uint8", torch::kUInt8},      {"torch.int8", torch::kInt8},
      {"torch.bfloat16", torch::kBFloat16}};

  auto it = dtype_map.find(dtype_str);
  if (it != dtype_map.end()) {
    return it->second;
  } else {
    throw std::invalid_argument("Unknown dtype string: " + dtype_str);
  }
}

// we need to reuse the cudamemory allocated , so we cannot release, when real_tensor free here
std::unordered_map<std::string, torch::Tensor> RestoreTensors2(
    const std::unordered_map<
        std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const std::unordered_map<int, void*>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        tensor_device_offsets) {
  std::unordered_map<std::string, torch::Tensor> state_dict;
  std::unordered_set<void*> handled_memory;
  for (const auto& [device, tensor_offset] : tensor_device_offsets) {
    for (const auto& p : tensor_offset) {
      std::string name = p.first;
      if (memory_base_address.find(device) != memory_base_address.end()) {
        void* base_address = memory_base_address.at(device);
        uint64_t offset = reinterpret_cast<uint64_t>(base_address) + p.second;

        torch::Device tensor_device(torch::kCUDA, device);
        auto [sizes, strides, type_str] = meta_state_dict.at(name);
        at::ScalarType dtype = stringToScalarType(type_str);
        // std::cerr << name << " " << sizes << " " << strides << " " << dtype
        // << std::endl;
        if (p.second == 0 &&
            handled_memory.find(base_address) == handled_memory.end()) {
          torch::Tensor real_tensor = torch::from_blob(
              reinterpret_cast<void*>(offset), c10::makeArrayRef(sizes),
              c10::makeArrayRef(strides), 
              // [](void* ptr) { cudaFree(ptr); },
              // don't release here, released outside
              [](void* ptr) {},
              torch::TensorOptions().device(tensor_device).dtype(dtype));
          state_dict[name] = real_tensor;
          handled_memory.insert(base_address);
          // std::cerr << "Tensor " << name << " is restored to device " <<
          // device << std::endl;
        } else {
          torch::Tensor real_tensor = torch::from_blob(
              reinterpret_cast<void*>(offset), sizes, strides, [](void* ptr) {},
              torch::TensorOptions().device(tensor_device).dtype(dtype));
          state_dict[name] = real_tensor;
        }
      } else {
        LOG(INFO) << "Cannot find device " << device;
        exit(1);
      }
    }
  }
  return state_dict;
}

std::unordered_map<std::string, torch::Tensor> RestoreTensors(
    const std::unordered_map<
        std::string, std::tuple<std::vector<int64_t>, std::vector<int64_t>,
                                std::string>>& meta_state_dict,
    const std::unordered_map<int, void*>& memory_base_address,
    const std::unordered_map<int, std::unordered_map<std::string, uint64_t>>&
        tensor_device_offsets) {
  std::unordered_map<std::string, torch::Tensor> state_dict;
  std::unordered_set<void*> handled_memory;
  for (const auto& [device, tensor_offset] : tensor_device_offsets) {
    for (const auto& p : tensor_offset) {
      std::string name = p.first;
      if (memory_base_address.find(device) != memory_base_address.end()) {
        void* base_address = memory_base_address.at(device);
        uint64_t offset = reinterpret_cast<uint64_t>(base_address) + p.second;

        torch::Device tensor_device(torch::kCUDA, device);
        auto [sizes, strides, type_str] = meta_state_dict.at(name);
        at::ScalarType dtype = stringToScalarType(type_str);
        // std::cerr << name << " " << sizes << " " << strides << " " << dtype
        // << std::endl;
        if (p.second == 0 &&
            handled_memory.find(base_address) == handled_memory.end()) {
          torch::Tensor real_tensor = torch::from_blob(
              reinterpret_cast<void*>(offset), c10::makeArrayRef(sizes),
              c10::makeArrayRef(strides), [](void* ptr) { cudaFree(ptr); },
              torch::TensorOptions().device(tensor_device).dtype(dtype));
          state_dict[name] = real_tensor;
          handled_memory.insert(base_address);
          // std::cerr << "Tensor " << name << " is restored to device " <<
          // device << std::endl;
        } else {
          torch::Tensor real_tensor = torch::from_blob(
              reinterpret_cast<void*>(offset), sizes, strides, [](void* ptr) {},
              torch::TensorOptions().device(tensor_device).dtype(dtype));
          state_dict[name] = real_tensor;
        }
      } else {
        LOG(INFO) << "Cannot find device " << device;
        exit(1);
      }
    }
  }
  return state_dict;
}

std::unordered_map<std::string, int> GetGpuUUID() {
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);  // Get the number of CUDA devices
  std::unordered_map<std::string, int> uuidToDeviceIdMap;

  for (int devId = 0; devId < deviceCount; ++devId) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, devId);  // Get properties for each device

    // Convert UUID bytes to string with unsigned char casting
    char uuidStr[80];
    snprintf(
        uuidStr, sizeof(uuidStr),
        "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
        (unsigned char)props.uuid.bytes[0], (unsigned char)props.uuid.bytes[1],
        (unsigned char)props.uuid.bytes[2], (unsigned char)props.uuid.bytes[3],
        (unsigned char)props.uuid.bytes[4], (unsigned char)props.uuid.bytes[5],
        (unsigned char)props.uuid.bytes[6], (unsigned char)props.uuid.bytes[7],
        (unsigned char)props.uuid.bytes[8], (unsigned char)props.uuid.bytes[9],
        (unsigned char)props.uuid.bytes[10],
        (unsigned char)props.uuid.bytes[11],
        (unsigned char)props.uuid.bytes[12],
        (unsigned char)props.uuid.bytes[13],
        (unsigned char)props.uuid.bytes[14],
        (unsigned char)props.uuid.bytes[15]);

    uuidToDeviceIdMap[std::string(uuidStr)] = devId;
  }

  return uuidToDeviceIdMap;
}

std::unordered_map<int, void*> AllocateCudaMemory(
    const std::unordered_map<int, size_t>& tensor_sizes) {
  std::unordered_map<int, void*> memory_ptrs;
  for (const auto& p : tensor_sizes) {
    int device = p.first;
    size_t size = p.second;
    void* ptr = nullptr;
    
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
      LOG(INFO) << "Failed to set CUDA device " << device << ": " 
                << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("Failed to set CUDA device");
    }
    
    err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess || ptr == nullptr) {
      LOG(INFO) << "Failed to allocate " << size << " bytes on device " << device 
                << ": " << cudaGetErrorString(err);
      throw std::runtime_error("CUDA memory allocation failed");
    }
    
    memory_ptrs[device] = ptr;
  }
  return memory_ptrs;
}

void FreeCudaMemory(
    const std::unordered_map<int, void*>& memory_ptrs) {
  for (const auto& p : memory_ptrs) {
    int device = p.first;
    void* ptr = p.second;
    cudaError_t err = cudaSetDevice(device);
    if (err != cudaSuccess) {
      LOG(INFO) << "Failed to set CUDA device " << device << ": " 
                << cudaGetErrorString(err) << std::endl;
      throw std::runtime_error("Failed to set CUDA device");
    }
    cudaFree(ptr);
  }
  return;
}

std::unordered_map<int, std::string> GetCudaMemoryHandles(
    const std::unordered_map<int, void*>& memory_ptrs) {
  std::unordered_map<int, std::string> memory_handles;
  for (const auto& p : memory_ptrs) {
    int device = p.first;
    void* ptr = p.second;
    cudaIpcMemHandle_t handle;
    cudaSetDevice(device);
    cudaIpcGetMemHandle(&handle, ptr);
    memory_handles[device] = std::string(reinterpret_cast<const char*>(&handle),
                                         sizeof(cudaIpcMemHandle_t));
  }
  return memory_handles;
}

std::unordered_map<int, std::vector<std::string>> GetCudaMemoryHandles(
    const std::unordered_map<int, std::vector<void*>>& memory_ptrs) {
  std::unordered_map<int, std::vector<std::string>> memory_handles;
  for (const auto& p : memory_ptrs) {
    auto device = p.first;
    const auto& ptrs = p.second;
    cudaIpcMemHandle_t handle;
    cudaSetDevice(device);

    std::vector<std::string> handles;
    for (const auto& ptr : ptrs) {
      cudaIpcGetMemHandle(&handle, ptr);
      handles.push_back(std::string(reinterpret_cast<const char*>(&handle),
                                    sizeof(cudaIpcMemHandle_t)));
    }
    memory_handles[device] = handles;
  }
  return memory_handles;
}

std::unordered_map<int, std::string> GetDeviceUuidMap() {
  std::unordered_map<std::string, int> gpu_uuid = GetGpuUUID();
  std::unordered_map<int, std::string> device_uuid_map;
  for (const auto& p : gpu_uuid) {
    if (device_uuid_map.find(p.second) != device_uuid_map.end()) {
      LOG(INFO) << "Duplicate device id: " << p.second;
      exit(1);
    }
    device_uuid_map[p.second] = p.first;
  }
  return device_uuid_map;
}

// 辅助函数：计算 tensor 跨越的 chunks
std::vector<std::tuple<int, size_t, size_t>> MapDataToChunks(
    size_t offset, size_t size, size_t chunk_size) {
  int start_chunk = offset / chunk_size;
  size_t offset_in_start_chunk = offset % chunk_size;
  size_t remaining_data = size;
  std::vector<std::tuple<int, size_t, size_t>> output;

  while (remaining_data > 0) {
    size_t chunk_data_size = (offset_in_start_chunk + remaining_data <= chunk_size)
                                 ? remaining_data
                                 : (chunk_size - offset_in_start_chunk);
    output.push_back(std::make_tuple(start_chunk, offset_in_start_chunk,
                                     chunk_data_size));
    remaining_data -= chunk_data_size;
    start_chunk++;
    offset_in_start_chunk = 0;
  }

  return output;
}

// 从共享内存构造基于 server 端管理的 pinned memory 的 CPU tensor
// memory_base_address: {chunk_id: base_ptr} - 每个 chunk 的基地址（通过共享内存或 mmap 获得）
// tensor_metadata: tensor 的元数据，包含 offset, size, shape, strides, dtype
// chunk_size: pinned memory 的 chunk 大小
std::unordered_map<std::string, torch::Tensor> RestoreTensorsFromPinnedMemory(
    const std::unordered_map<int, void*>& memory_base_address,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size) {
  std::unordered_map<std::string, torch::Tensor> state_dict;

  // 正则表达式匹配 experts 下 w1, w2, w3 相关的 tensor
  std::regex expert_weight_pattern(
      R"(model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123])");

  for (const auto& [name, info] : tensor_metadata) {
    // 只处理 experts 下 w1, w2, w3 相关的 tensor
    if (!std::regex_search(name, expert_weight_pattern)) {
      continue;
    }

    // 从 tensor_metadata 获取 offset, size, shape, strides, dtype
    auto [offset, tensor_size_bytes, shape, strides, dtype_str] = info;

    // 转换 shape 和 strides 为 int64_t 向量
    std::vector<int64_t> sizes_int64;
    for (size_t s : shape) {
      sizes_int64.push_back(static_cast<int64_t>(s));
    }

    std::vector<int64_t> strides_int64;
    for (size_t s : strides) {
      strides_int64.push_back(static_cast<int64_t>(s));
    }

    at::ScalarType dtype = stringToScalarType(dtype_str);

    // 使用 MapDataToChunks 计算 tensor 跨越哪些 chunks
    std::vector<std::tuple<int, size_t, size_t>> chunks =
        MapDataToChunks(offset, tensor_size_bytes, chunk_size);

    if (chunks.empty()) {
      LOG(INFO) << "Failed to map tensor " << name << " to chunks";
      continue;
    }

    // 检查 tensor 是否完全在一个 chunk 内
    if (chunks.size() == 1) {
      // Tensor 完全在一个 chunk 内，可以直接使用 torch::from_blob
      auto [chunk_id, offset_in_chunk, chunk_data_size] = chunks[0];

      // 从 memory_base_address 获取该 chunk 的基地址
      if (memory_base_address.find(chunk_id) == memory_base_address.end()) {
        LOG(INFO) << "Tensor " << name << " chunk_id " << chunk_id
                   << " not found in memory_base_address";
        continue;
      }

      void* chunk_base = memory_base_address.at(chunk_id);
      void* tensor_data_ptr =
          static_cast<char*>(chunk_base) + offset_in_chunk;

      // 从共享内存创建 tensor
      torch::Tensor tensor = torch::from_blob(
          tensor_data_ptr, c10::makeArrayRef(sizes_int64),
          c10::makeArrayRef(strides_int64),
          [](void* ptr) {},  // 不释放内存，由 server 端管理
          torch::TensorOptions().device(torch::kCPU).dtype(dtype));

      state_dict[name] = tensor;
    } else {
      // Tensor 跨越多个 chunks，chunks 在内存中不连续
      // PyTorch 的 tensor 底层需要连续内存，torch::from_blob 要求连续内存
      // 对于跨 chunks 的情况，无法直接创建 tensor，报错并跳过
      LOG(INFO) << "Tensor " << name << " spans " << chunks.size()
                 << " non-contiguous chunks (offset=" << offset
                 << ", size=" << tensor_size_bytes
                 << "). Cannot create tensor from non-contiguous memory.";
      continue;
    }
  }

  LOG(INFO) << "Created " << state_dict.size()
            << " expert tensors from shared pinned memory";
  return state_dict;
}

// 从共享内存名称列表创建 CPU tensor
// shm_names: 共享内存名称列表
// tensor_metadata: tensor 的元数据，包含 offset, size, shape, strides, dtype
// chunk_size: pinned memory 的 chunk 大小
std::unordered_map<std::string, torch::Tensor> RestoreTensorsFromSharedMemoryNames(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size) {
  std::unordered_map<std::string, torch::Tensor> state_dict;

  if (shm_names.empty()) {
    LOG(INFO) << "shm_names is empty";
    return {};
  }

  // 分配连续的虚拟地址空间来映射所有共享内存块
  size_t total_size = shm_names.size() * chunk_size;
  void* contiguous_memory = nullptr;
  
  // 使用 mmap 预留连续的虚拟地址空间（MAP_ANONYMOUS 不分配物理内存）
  contiguous_memory = mmap(nullptr, total_size, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (contiguous_memory == MAP_FAILED) {
    LOG(INFO) << "Failed to reserve contiguous virtual address space of size " << total_size
               << ": " << strerror(errno);
    return {};
  }

  // 将每个共享内存对象直接映射到连续虚拟地址空间的对应位置
  std::unordered_map<int, void*> memory_base_address;
  std::vector<int> shm_fds;  // 用于错误处理时清理
  
  for (size_t chunk_id = 0; chunk_id < shm_names.size(); ++chunk_id) {
    const std::string& shm_name = shm_names[chunk_id];
    
    // 打开共享内存对象
    int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0);
    if (shm_fd == -1) {
      LOG(INFO) << "Failed to open shared memory " << shm_name << ": "
                 << strerror(errno);
      // Clean up already mapped memory
      for (size_t i = 0; i < chunk_id; ++i) {
        void* chunk_ptr = static_cast<char*>(contiguous_memory) + i * chunk_size;
        munmap(chunk_ptr, chunk_size);
      }
      munmap(contiguous_memory, total_size);
      return {};
    }
    shm_fds.push_back(shm_fd);
    
    // 计算该 chunk 在连续地址空间中的位置
    void* chunk_addr = static_cast<char*>(contiguous_memory) + chunk_id * chunk_size;
    
    // 使用 MAP_FIXED 将共享内存映射到连续地址空间的指定位置
    void* mapped_addr = mmap(chunk_addr, chunk_size, PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_FIXED, shm_fd, 0);
    close(shm_fd);  // 关闭文件描述符，mmap 保持映射
    
    if (mapped_addr == MAP_FAILED || mapped_addr != chunk_addr) {
      LOG(INFO) << "Failed to mmap shared memory " << shm_name
                 << " to contiguous address " << chunk_addr << ": "
                 << strerror(errno);
      // Clean up already mapped memory
      for (size_t i = 0; i < chunk_id; ++i) {
        void* chunk_ptr = static_cast<char*>(contiguous_memory) + i * chunk_size;
        munmap(chunk_ptr, chunk_size);
      }
      munmap(contiguous_memory, total_size);
      return {};
    }
    
    // 更新 memory_base_address 指向连续地址空间中的位置
    memory_base_address[chunk_id] = mapped_addr;
    
    LOG(INFO) << "Mapped shared memory " << shm_name << " to contiguous address "
              << mapped_addr << " (chunk " << chunk_id << ")";
  }

  // 直接使用连续地址空间创建 tensors
  // 由于内存现在是连续的，即使 tensor 跨越多个 chunks，在连续地址空间中也是连续的
  for (const auto& [name, info] : tensor_metadata) {

    // 从 tensor_metadata 获取 offset, size, shape, strides, dtype
    auto [offset, tensor_size_bytes, shape, strides, dtype_str] = info;

    // 转换 shape 和 strides 为 int64_t 向量
    std::vector<int64_t> sizes_int64;
    for (size_t s : shape) {
      sizes_int64.push_back(static_cast<int64_t>(s));
    }

    std::vector<int64_t> strides_int64;
    for (size_t s : strides) {
      strides_int64.push_back(static_cast<int64_t>(s));
    }

    at::ScalarType dtype = stringToScalarType(dtype_str);

    // 计算 tensor 在连续地址空间中的位置
    // 由于内存是连续的，可以直接使用 offset 计算地址
    void* tensor_data_ptr = static_cast<char*>(contiguous_memory) + offset;

    // 验证 tensor 是否在有效范围内
    if (offset + tensor_size_bytes > total_size) {
      LOG(INFO) << "Tensor " << name << " offset " << offset
                 << " + size " << tensor_size_bytes
                 << " exceeds total size " << total_size;
      continue;
    }

    // 从连续地址空间创建 tensor
    // 由于内存是连续的，即使 tensor 跨越多个 chunks 也可以直接创建
    torch::Tensor tensor = torch::from_blob(
        tensor_data_ptr, c10::makeArrayRef(sizes_int64),
        c10::makeArrayRef(strides_int64),
        [](void* ptr) {},  // 临时 deleter，稍后会替换
        torch::TensorOptions().device(torch::kCPU).dtype(dtype));

    state_dict[name] = tensor;
  }

  // 如果没有创建任何 tensor，释放所有映射并返回
  if (state_dict.empty()) {
    LOG(INFO) << "No tensors created from shared memory, releasing all mappings";
    for (size_t i = 0; i < shm_names.size(); ++i) {
      void* chunk_ptr = static_cast<char*>(contiguous_memory) + i * chunk_size;
      munmap(chunk_ptr, chunk_size);
    }
    munmap(contiguous_memory, total_size);
    return {};
  }

  // 使用引用计数来管理连续内存的生命周期
  // 所有 tensor 共享同一块连续内存，只有当所有 tensor 都被销毁时才释放
  static std::unordered_map<void*, std::shared_ptr<std::atomic<int>>> memory_ref_counts;
  static std::mutex ref_count_mutex;
  
  std::lock_guard<std::mutex> lock(ref_count_mutex);
  auto ref_count = std::make_shared<std::atomic<int>>(state_dict.size());
  memory_ref_counts[contiguous_memory] = ref_count;
  
  // 为每个 tensor 设置自定义 deleter，使用引用计数管理内存
  for (auto& [name, tensor] : state_dict) {
    // 获取 tensor 的底层数据指针
    void* data_ptr = tensor.data_ptr();
    
    // 创建一个新的 tensor，使用自定义 deleter
    torch::Tensor new_tensor = torch::from_blob(
        data_ptr,
        tensor.sizes(),
        tensor.strides(),
        [ref_count, contiguous_memory, total_size, chunk_size, num_chunks = shm_names.size()](void* ptr) {
          // 减少引用计数
          int remaining = ref_count->fetch_sub(1) - 1;
          if (remaining == 0) {
            // 最后一个 tensor 被销毁，释放所有映射的共享内存
            for (size_t i = 0; i < num_chunks; ++i) {
              void* chunk_ptr = static_cast<char*>(contiguous_memory) + i * chunk_size;
              munmap(chunk_ptr, chunk_size);
            }
            // 释放预留的虚拟地址空间
            munmap(contiguous_memory, total_size);
            std::lock_guard<std::mutex> lock(ref_count_mutex);
            memory_ref_counts.erase(contiguous_memory);
          }
        },
        tensor.options());
    
    state_dict[name] = new_tensor;
  }

  LOG(INFO) << "Created " << state_dict.size()
            << " tensors from shared memory mapped to contiguous address space ("
            << (total_size / 1024 / 1024) << " MB)";

  return state_dict;
}

//将 一个layer的若干个专家映射到连续虚拟空间中
std::unordered_map<std::string, torch::Tensor> RestoreExpertsFromSharedMemory(
  const std::vector<std::string>& shm_names,
  const TensorIndexResizeMap& tensor_metadata,
  size_t chunk_size,
  const std::vector<std::string>& name_continuous_space
) {
  auto start_time = std::chrono::high_resolution_clock::now();
  std::unordered_map<std::string, torch::Tensor> state_dict;

  if (shm_names.empty()) {
    LOG(INFO) << "shm_names is empty";
    return {};
  }

  if (name_continuous_space.empty()) {
    LOG(INFO) << "name_continuous_space is empty";
    return {};
  }

  // 创建名称集合以便快速查找
  std::unordered_set<std::string> name_set(name_continuous_space.begin(), name_continuous_space.end());

  // 获取页大小（用于内存对齐）
  const size_t page_size = sysconf(_SC_PAGESIZE);

  // 第一步：计算 name_continuous_space 所需的连续内存大小
  // 收集所有需要的 tensor 信息，并计算总大小
  struct TensorInfo {
    std::string name;
    uint64_t offset;
    uint64_t size;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::string dtype;
    size_t chunk_id;  // 所在的 chunk ID
    size_t chunk_offset;  // 在 chunk 中的偏移
    bool crosses_chunk_boundary;  // 是否跨越 chunk 边界
    size_t end_chunk_id;  // 结束的 chunk ID
  };
  
  std::vector<TensorInfo> tensor_infos;
  size_t total_continuous_size = 0;
  
  auto step1_start = std::chrono::high_resolution_clock::now();
  // 找到所有需要的 tensor 并计算总大小
  for (const std::string& name : name_continuous_space) {
    auto it = tensor_metadata.find(name);
    if (it == tensor_metadata.end()) {
      LOG(INFO) << "Tensor " << name << " not found in tensor_metadata";
      continue;
    }
    
    auto [offset, tensor_size_bytes, shape, strides, dtype_str] = it->second;
    
    // 计算 tensor 所在的 chunk 和在 chunk 中的偏移
    size_t chunk_id = offset / chunk_size;
    size_t chunk_offset = offset % chunk_size;
    size_t end_chunk_id = (offset + tensor_size_bytes - 1) / chunk_size;
    bool crosses_chunk_boundary = (chunk_id != end_chunk_id);
    
    TensorInfo info;
    info.name = name;
    info.offset = offset;
    info.size = tensor_size_bytes;
    info.shape = shape;
    info.strides = strides;
    info.dtype = dtype_str;
    info.chunk_id = chunk_id;
    info.chunk_offset = chunk_offset;
    info.crosses_chunk_boundary = crosses_chunk_boundary;
    info.end_chunk_id = end_chunk_id;
    
    tensor_infos.push_back(info);
    total_continuous_size += tensor_size_bytes;
  }
  
  if (tensor_infos.empty()) {
    LOG(INFO) << "No valid tensors found in name_continuous_space";
    return {};
  }
  
  auto step1_end = std::chrono::high_resolution_clock::now();
  auto step1_duration = std::chrono::duration_cast<std::chrono::microseconds>(step1_end - step1_start);
  LOG(INFO) << "[RestoreExpertsFromSharedMemory] Step1: Collect tensor info cost " 
            << step1_duration.count() / 1000.0 << " ms";
  
  // 对齐到页边界（通常是 4096 字节）
  total_continuous_size = ((total_continuous_size + page_size - 1) / page_size) * page_size;
  
  // 分配连续的虚拟地址空间来存储所有需要的 tensor
  void* contiguous_memory = nullptr;
  
  auto step2_start = std::chrono::high_resolution_clock::now();
  // 使用 mmap 分配连续的虚拟地址空间（MAP_ANONYMOUS 分配可写内存）
  contiguous_memory = mmap(nullptr, total_continuous_size, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (contiguous_memory == MAP_FAILED) {
    LOG(INFO) << "Failed to allocate contiguous virtual address space of size " 
              << total_continuous_size << ": " << strerror(errno);
    return {};
  }
  auto step2_end = std::chrono::high_resolution_clock::now();
  auto step2_duration = std::chrono::duration_cast<std::chrono::microseconds>(step2_end - step2_start);
  LOG(INFO) << "[RestoreExpertsFromSharedMemory] Step2: Allocate contiguous memory (" 
            << total_continuous_size / 1024 / 1024 << " MB) cost " 
            << step2_duration.count() / 1000.0 << " ms";

  // 第二步：计算每个 tensor 在连续地址空间中的目标位置
  auto step3_start = std::chrono::high_resolution_clock::now();
  size_t current_offset = 0;
  std::unordered_map<std::string, size_t> tensor_offsets;  // 记录每个 tensor 在连续空间中的偏移
  
  for (const auto& info : tensor_infos) {
    tensor_offsets[info.name] = current_offset;
    current_offset += info.size;
  }
  auto step3_end = std::chrono::high_resolution_clock::now();
  auto step3_duration = std::chrono::duration_cast<std::chrono::microseconds>(step3_end - step3_start);
  LOG(INFO) << "[RestoreExpertsFromSharedMemory] Step3: Calculate tensor offsets cost " 
            << step3_duration.count() / 1000.0 << " ms";

  // 第三步：直接从共享内存映射到连续地址空间（避免复制）
  // 使用 mmap 的 MAP_FIXED 选项将共享内存的特定部分映射到连续地址空间
  auto step4_start = std::chrono::high_resolution_clock::now();
  std::unordered_map<size_t, int> chunk_fds;  // 保存每个 chunk 的文件描述符，稍后关闭
  
  // 记录已映射的区域，避免重复映射
  struct MappedRegion {
    void* addr;
    size_t size;
  };
  std::vector<MappedRegion> mapped_regions;
  
  for (const auto& info : tensor_infos) {
    if (info.chunk_id >= shm_names.size()) {
      LOG(INFO) << "Chunk ID " << info.chunk_id << " exceeds shm_names size";
      // Clean up
      for (auto& [chunk_id, fd] : chunk_fds) {
        close(fd);
      }
      for (const auto& region : mapped_regions) {
        munmap(region.addr, region.size);
      }
      munmap(contiguous_memory, total_continuous_size);
      return {};
    }
    
    // 获取或打开共享内存文件描述符
    int shm_fd;
    if (chunk_fds.find(info.chunk_id) == chunk_fds.end()) {
      const std::string& shm_name = shm_names[info.chunk_id];
      shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0);
      if (shm_fd == -1) {
        LOG(INFO) << "Failed to open shared memory " << shm_name << ": "
                   << strerror(errno);
        // Clean up
        for (auto& [cid, fd] : chunk_fds) {
          close(fd);
        }
        for (const auto& region : mapped_regions) {
          munmap(region.addr, region.size);
        }
        munmap(contiguous_memory, total_continuous_size);
        return {};
      }
      chunk_fds[info.chunk_id] = shm_fd;
    } else {
      shm_fd = chunk_fds[info.chunk_id];
    }
    
    // 计算 tensor 在连续地址空间中的目标位置
    size_t continuous_offset = tensor_offsets[info.name];
    void* target_addr = static_cast<char*>(contiguous_memory) + continuous_offset;
    
    // 验证目标地址在连续地址空间范围内
    if (target_addr < contiguous_memory || 
        static_cast<char*>(target_addr) + info.size > static_cast<char*>(contiguous_memory) + total_continuous_size) {
      LOG(INFO) << "Tensor " << info.name << " target address out of range";
      continue;
    }
    
    // 关键修复：如果 tensor 跨越 chunk 边界，必须使用 memcpy 复制数据
    // 因为 mmap 一次只能映射一个 chunk，无法处理跨 chunk 的情况
    if (info.crosses_chunk_boundary) {
      // 跨 chunk 的 tensor，使用 memcpy 复制数据
      size_t total_copied = 0;
      size_t remaining_size = info.size;
      size_t current_chunk_id = info.chunk_id;
      size_t current_chunk_offset = info.chunk_offset;
      
      while (remaining_size > 0) {
        // 计算当前 chunk 中可复制的数据量
        size_t chunk_remaining = chunk_size - current_chunk_offset;
        size_t copy_size = (remaining_size < chunk_remaining) ? remaining_size : chunk_remaining;
        
        // 打开当前 chunk 的共享内存
        int shm_fd_current;
        if (chunk_fds.find(current_chunk_id) == chunk_fds.end()) {
          if (current_chunk_id >= shm_names.size()) {
            LOG(INFO) << "[RestoreExpertsFromSharedMemory] ERROR: Chunk ID " << current_chunk_id 
                      << " exceeds shm_names size " << shm_names.size();
            break;
          }
          const std::string& shm_name = shm_names[current_chunk_id];
          shm_fd_current = shm_open(shm_name.c_str(), O_RDWR, 0);
          if (shm_fd_current == -1) {
            LOG(INFO) << "[RestoreExpertsFromSharedMemory] ERROR: Failed to open shared memory " 
                      << shm_name << ": " << strerror(errno);
            break;
          }
          chunk_fds[current_chunk_id] = shm_fd_current;
        } else {
          shm_fd_current = chunk_fds[current_chunk_id];
        }
        
        // 映射当前 chunk 的数据
        off_t aligned_chunk_offset = (current_chunk_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
        off_t chunk_offset_adjustment = current_chunk_offset - aligned_chunk_offset;
        size_t map_size = ((copy_size + chunk_offset_adjustment + page_size - 1) / page_size) * page_size;
        
        void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd_current, aligned_chunk_offset);
        if (temp_mapped_addr == MAP_FAILED) {
          LOG(INFO) << "[RestoreExpertsFromSharedMemory] ERROR: Failed to mmap temp for cross-chunk tensor " 
                    << info.name << " chunk " << current_chunk_id << ": " << strerror(errno);
          break;
        }
        
        // 复制数据
        void* src_addr = static_cast<char*>(temp_mapped_addr) + chunk_offset_adjustment;
        void* dst_addr = static_cast<char*>(target_addr) + total_copied;
        std::memcpy(dst_addr, src_addr, copy_size);
        
        // 取消映射
        munmap(temp_mapped_addr, map_size);
        
        total_copied += copy_size;
        remaining_size -= copy_size;
        current_chunk_id++;
        current_chunk_offset = 0;
      }
      
      if (total_copied != info.size) {
        LOG(INFO) << "[RestoreExpertsFromSharedMemory] ERROR: Failed to copy cross-chunk tensor " 
                  << info.name << ": copied " << total_copied << " / " << info.size << " bytes";
        continue;
      }
      
      LOG(INFO) << "[RestoreExpertsFromSharedMemory] Tensor " << info.name
                << " copied via memcpy (crosses chunks " << info.chunk_id 
                << " -> " << info.end_chunk_id << ")"
                << " target_addr=" << target_addr
                << " size=" << info.size;
    } else {
      // 不跨 chunk 的 tensor，使用现有的映射逻辑
    // 计算共享内存中的源位置（页对齐）
    // mmap 的 offset 必须是页大小的倍数
    off_t shm_offset = static_cast<off_t>(info.chunk_offset);
    off_t aligned_shm_offset = (shm_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
    off_t offset_adjustment = shm_offset - aligned_shm_offset;
    
    // 计算需要映射的大小（页对齐，包含调整部分）
    size_t map_size = ((info.size + offset_adjustment + page_size - 1) / page_size) * page_size;
    
    // 调整目标地址以匹配页对齐（但确保不超出范围）
    void* aligned_target_addr = static_cast<char*>(target_addr) - offset_adjustment;
    
    // 检查调整后的地址是否在范围内
    if (aligned_target_addr < contiguous_memory) {
      // 如果调整后超出下界，需要调整映射策略
      // 这种情况下，我们可能需要复制数据，或者调整连续地址空间的布局
      LOG(INFO) << "Warning: Tensor " << info.name 
                 << " requires alignment adjustment that would exceed contiguous memory bounds. "
                 << "Consider aligning tensors to page boundaries.";
      // 对于这种情况，我们仍然尝试映射，但可能需要特殊处理
      aligned_target_addr = contiguous_memory;
      aligned_shm_offset = 0;
      offset_adjustment = 0;
      map_size = ((info.size + page_size - 1) / page_size) * page_size;
    }
    
    // 检查是否已经映射了这个区域（避免重复映射）
    bool already_mapped = false;
    for (const auto& region : mapped_regions) {
      if (aligned_target_addr >= region.addr && 
          aligned_target_addr < static_cast<char*>(region.addr) + region.size) {
        already_mapped = true;
        break;
      }
    }
    
    if (!already_mapped) {
      // 使用 MAP_FIXED 直接将共享内存映射到连续地址空间
      void* mapped_addr = mmap(aligned_target_addr, map_size, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_FIXED, shm_fd, aligned_shm_offset);
      
      if (mapped_addr == MAP_FAILED || mapped_addr != aligned_target_addr) {
        LOG(INFO) << "Failed to mmap shared memory chunk " << info.chunk_id
                   << " offset " << aligned_shm_offset << " to contiguous address "
                   << aligned_target_addr << ": " << strerror(errno);
        // Clean up
        for (auto& [cid, fd] : chunk_fds) {
          close(fd);
        }
        for (const auto& region : mapped_regions) {
          munmap(region.addr, region.size);
        }
        munmap(contiguous_memory, total_continuous_size);
        return {};
      }
      
      mapped_regions.push_back({aligned_target_addr, map_size});
      }
    }
  }
  
  // 关闭所有文件描述符（mmap 保持映射）
  for (auto& [chunk_id, fd] : chunk_fds) {
    close(fd);
  }
  auto step4_end = std::chrono::high_resolution_clock::now();
  auto step4_duration = std::chrono::duration_cast<std::chrono::microseconds>(step4_end - step4_start);
  LOG(INFO) << "[RestoreExpertsFromSharedMemory] Step4: Map shared memory to contiguous space cost " 
            << step4_duration.count() / 1000.0 << " ms";

  // 第四步：在连续地址空间中创建 tensor（使用新的连续偏移）
  auto step5_start = std::chrono::high_resolution_clock::now();
  // 为了满足 test_einsum_experts.py 的需求，我们需要让相同类型的 tensor 共享底层存储
  // 方法：创建一个大的 tensor，然后用 view 来分割，这样它们就共享同一个底层存储
  
  // 首先，按类型分组 tensor（假设相同形状的 tensor 是同一类型）
  std::unordered_map<std::string, std::vector<std::pair<std::string, TensorInfo>>> type_groups;
  for (const auto& info : tensor_infos) {
    // 使用 shape 和 dtype 作为类型标识
    std::string type_key = info.dtype;
    for (size_t s : info.shape) {
      type_key += "_" + std::to_string(s);
    }
    type_groups[type_key].push_back({info.name, info});
  }
  
  // 为每个类型组创建共享存储的 tensor
  for (const auto& [type_key, group] : type_groups) {
    if (group.empty()) continue;
    
    const auto& first_info = group[0].second;
    
    // 检查是否所有 tensor 形状相同（同一类型）
    bool same_shape = true;
    for (const auto& [name, info] : group) {
      if (info.shape != first_info.shape || info.dtype != first_info.dtype) {
        same_shape = false;
        break;
      }
    }
    
    if (same_shape && group.size() > 1) {
      // 所有 tensor 形状相同，创建一个连续的大 tensor
      // 找到第一个 tensor 在连续地址空间中的位置
      size_t first_offset = tensor_offsets[group[0].first];
      void* group_base_ptr = static_cast<char*>(contiguous_memory) + first_offset;
      
      // 创建一个大 tensor，包含所有相同类型的 tensor
      // 形状为 [N, ...]，其中 N 是 tensor 数量
      std::vector<int64_t> group_shape;
      group_shape.push_back(static_cast<int64_t>(group.size()));
      for (size_t s : first_info.shape) {
        group_shape.push_back(static_cast<int64_t>(s));
      }
      
      // 计算 strides（连续存储）
      std::vector<int64_t> group_strides;
      int64_t stride = 1;
      for (int i = static_cast<int>(group_shape.size()) - 1; i >= 0; --i) {
        group_strides.insert(group_strides.begin(), stride);
        stride *= group_shape[i];
      }
      
      at::ScalarType dtype = stringToScalarType(first_info.dtype);
      
      // 创建连续的大 tensor（所有相同形状的 tensor 共享这个底层存储）
      // 这是一个 [N, ...] 形状的大 tensor，其中 N 是 tensor 数量
      torch::Tensor group_tensor = torch::from_blob(
          group_base_ptr, c10::makeArrayRef(group_shape),
          c10::makeArrayRef(group_strides),
          [](void* ptr) {},  // 临时 deleter，稍后会替换
          torch::TensorOptions().device(torch::kCPU).dtype(dtype));
      
      // 将 big_tensor 添加到 state_dict 中，使用第一个 tensor 的名称 + "_big_tensor" 作为 key
      // Python 端可以直接使用这个 big_tensor，无需再使用 as_strided
      std::string big_tensor_key = "big_tensor";
      state_dict[big_tensor_key] = group_tensor;
      
      // 为每个 tensor 创建 view（都共享同一个 group_tensor 的底层存储）
      // 使用索引操作 group_tensor[i] 会自动创建共享底层存储的 view
      // 这样所有 tensor 都共享同一个连续的大存储
      for (size_t i = 0; i < group.size(); ++i) {
        const std::string& name = group[i].first;
        // group_tensor[i] 创建一个 view，共享同一个大的底层存储
        torch::Tensor tensor_view = group_tensor[i];
        state_dict[name] = tensor_view;
      }
    } else {
      // 形状不同或只有一个 tensor，单独创建
      for (const auto& [name, info] : group) {
        std::vector<int64_t> sizes_int64;
        for (size_t s : info.shape) {
          sizes_int64.push_back(static_cast<int64_t>(s));
        }

        std::vector<int64_t> strides_int64;
        for (size_t s : info.strides) {
          strides_int64.push_back(static_cast<int64_t>(s));
        }

        at::ScalarType dtype = stringToScalarType(info.dtype);

        size_t continuous_offset = tensor_offsets[name];
        void* tensor_data_ptr = static_cast<char*>(contiguous_memory) + continuous_offset;

        torch::Tensor tensor = torch::from_blob(
            tensor_data_ptr, c10::makeArrayRef(sizes_int64),
            c10::makeArrayRef(strides_int64),
            [](void* ptr) {},
            torch::TensorOptions().device(torch::kCPU).dtype(dtype));

        state_dict[name] = tensor;
      }
    }
  }

  // 如果没有创建任何 tensor，释放连续内存并返回
  if (state_dict.empty()) {
    LOG(INFO) << "No tensors created from shared memory, releasing contiguous memory";
    munmap(contiguous_memory, total_continuous_size);
    return {};
  }
  auto step5_end = std::chrono::high_resolution_clock::now();
  auto step5_duration = std::chrono::duration_cast<std::chrono::microseconds>(step5_end - step5_start);
  LOG(INFO) << "[RestoreExpertsFromSharedMemory] Step5: Create tensors cost " 
            << step5_duration.count() / 1000.0 << " ms";

  // 使用引用计数来管理连续内存的生命周期
  // 所有 tensor 共享同一块连续内存，只有当所有 tensor 都被销毁时才释放
  auto step6_start = std::chrono::high_resolution_clock::now();
  static std::unordered_map<void*, std::shared_ptr<std::atomic<int>>> memory_ref_counts;
  static std::mutex ref_count_mutex;
  
  std::lock_guard<std::mutex> lock(ref_count_mutex);
  auto ref_count = std::make_shared<std::atomic<int>>(state_dict.size());
  memory_ref_counts[contiguous_memory] = ref_count;

  // 为每个 tensor 设置自定义 deleter，使用引用计数管理内存
  // 注意：相同形状的 tensor 已经共享同一个底层存储（group_tensor），
  // 这里只是为每个 tensor view 设置 deleter，确保内存正确释放
  for (auto& [name, tensor] : state_dict) {
    // 获取 tensor 的底层数据指针和存储信息
    void* data_ptr = tensor.data_ptr();
    
    // 创建一个新的 tensor view，使用自定义 deleter
    // 这样即使 tensor view 被销毁，只要还有其他 tensor 在使用，内存就不会被释放
    torch::Tensor new_tensor = torch::from_blob(
        data_ptr,
        tensor.sizes(),
        tensor.strides(),
        [ref_count, contiguous_memory, total_continuous_size](void* ptr) {
          // 减少引用计数
          int remaining = ref_count->fetch_sub(1) - 1;
          if (remaining == 0) {
            // 最后一个 tensor 被销毁，释放连续内存
            munmap(contiguous_memory, total_continuous_size);
            std::lock_guard<std::mutex> lock(ref_count_mutex);
            memory_ref_counts.erase(contiguous_memory);
          }
        },
        tensor.options());
    
    state_dict[name] = new_tensor;
  }
  auto step6_end = std::chrono::high_resolution_clock::now();
  auto step6_duration = std::chrono::duration_cast<std::chrono::microseconds>(step6_end - step6_start);
  LOG(INFO) << "[RestoreExpertsFromSharedMemory] Step6: Setup reference counting cost " 
            << step6_duration.count() / 1000.0 << " ms";

  LOG(INFO) << "Created " << state_dict.size()
            << " expert tensors from shared memory in contiguous address space ("
            << (total_continuous_size / 1024 / 1024) << " MB). "
            << "Tensors are arranged contiguously to avoid copying during einsum operations.";

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - start_time);
  LOG(INFO) << "[RestoreExpertsFromSharedMemory] Total cost " 
            << total_duration.count() / 1000.0 << " ms";

  return state_dict;
}

// 支持多个 vector，为每个 vector 创建一个 big_tensor
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemory(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups
) {
  std::unordered_map<std::string, torch::Tensor> state_dict;

  if (shm_names.empty()) {
    LOG(INFO) << "shm_names is empty";
    return {};
  }

  if (name_groups.empty()) {
    LOG(INFO) << "name_groups is empty";
    return {};
  }

  const size_t page_size = sysconf(_SC_PAGESIZE);

  // Tensor 信息结构
  struct TensorInfo {
    std::string name;
    uint64_t offset;
    uint64_t size;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::string dtype;
    size_t chunk_id;
    size_t chunk_offset;
    size_t group_id;
    bool crosses_chunk_boundary;
    size_t end_chunk_id;
  };

  // Group 信息结构
  struct GroupInfo {
    size_t group_id;
    std::vector<TensorInfo> tensors;
    size_t total_size;
    void* contiguous_memory;
    std::vector<int64_t> big_tensor_shape;
    std::vector<int64_t> big_tensor_strides;
    at::ScalarType dtype;
  };

  std::vector<GroupInfo> groups_info;

  // 第一步：收集每个 group 的 tensor 信息
  for (size_t group_id = 0; group_id < name_groups.size(); ++group_id) {
    const auto& name_group = name_groups[group_id];
    
    if (name_group.empty()) {
      LOG(INFO) << "Group " << group_id << " is empty, skipping";
      continue;
    }

    GroupInfo group_info;
    group_info.group_id = group_id;
    group_info.total_size = 0;
    group_info.contiguous_memory = nullptr;

    for (const std::string& name : name_group) {
      auto it = tensor_metadata.find(name);
      if (it == tensor_metadata.end()) {
        LOG(INFO) << "Tensor " << name << " not found in tensor_metadata";
        continue;
      }

      auto [offset, tensor_size_bytes, shape, strides, dtype_str] = it->second;

      size_t chunk_id = offset / chunk_size;
      size_t chunk_offset = offset % chunk_size;
      size_t end_chunk_id = (offset + tensor_size_bytes - 1) / chunk_size;
      bool crosses_chunk_boundary = (chunk_id != end_chunk_id);

      TensorInfo info;
      info.name = name;
      info.offset = offset;
      info.size = tensor_size_bytes;
      info.shape = shape;
      info.strides = strides;
      info.dtype = dtype_str;
      info.chunk_id = chunk_id;
      info.chunk_offset = chunk_offset;
      info.group_id = group_id;
      info.crosses_chunk_boundary = crosses_chunk_boundary;
      info.end_chunk_id = end_chunk_id;

      group_info.tensors.push_back(info);
      group_info.total_size += tensor_size_bytes;
    }

    if (group_info.tensors.empty()) {
      LOG(INFO) << "Group " << group_id << " has no valid tensors, skipping";
      continue;
    }

    const auto& first_tensor = group_info.tensors[0];
    bool same_shape = true;
    for (const auto& tensor_info : group_info.tensors) {
      if (tensor_info.shape != first_tensor.shape || tensor_info.dtype != first_tensor.dtype) {
        same_shape = false;
        break;
      }
    }

    if (!same_shape) {
      LOG(INFO) << "Group " << group_id << " has tensors with different shapes or dtypes, skipping";
      continue;
    }

    group_info.big_tensor_shape.push_back(static_cast<int64_t>(group_info.tensors.size()));
    for (size_t s : first_tensor.shape) {
      group_info.big_tensor_shape.push_back(static_cast<int64_t>(s));
    }

    int64_t stride = 1;
    for (int i = static_cast<int>(group_info.big_tensor_shape.size()) - 1; i >= 0; --i) {
      group_info.big_tensor_strides.insert(group_info.big_tensor_strides.begin(), stride);
      stride *= group_info.big_tensor_shape[i];
    }

    group_info.dtype = stringToScalarType(first_tensor.dtype);
    group_info.total_size = ((group_info.total_size + page_size - 1) / page_size) * page_size;

    groups_info.push_back(group_info);
  }

  if (groups_info.empty()) {
    LOG(INFO) << "No valid groups found";
    return {};
  }

  // 第二步：为每个 group 分配连续的虚拟地址空间
  for (auto& group_info : groups_info) {
    void* contiguous_memory = mmap(nullptr, group_info.total_size, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (contiguous_memory == MAP_FAILED) {
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Failed to allocate contiguous virtual address space for group "
                << group_info.group_id << " of size " << group_info.total_size
                << ": " << strerror(errno);
      for (auto& prev_group : groups_info) {
        if (prev_group.contiguous_memory != nullptr) {
          munmap(prev_group.contiguous_memory, prev_group.total_size);
        }
      }
      return {};
    }
    group_info.contiguous_memory = contiguous_memory;
    
    // 检查地址是否页对齐（页大小通常是 4096 = 0x1000）
    size_t page_size = 4096;
    uintptr_t addr_val = reinterpret_cast<uintptr_t>(contiguous_memory);
    if (addr_val % page_size != 0) {
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] WARNING: Group " << group_info.group_id
                << " contiguous_memory=" << contiguous_memory 
                << " (0x" << std::hex << addr_val << std::dec << ") is not page-aligned!"
                << " Remainder: " << (addr_val % page_size);
    }
    
    LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Allocated contiguous memory for group " 
              << group_info.group_id << " at " << contiguous_memory 
              << " (0x" << std::hex << addr_val << std::dec << ") size=" 
              << group_info.total_size << " bytes";
  }

  // 第三步：为每个 group 映射数据
  std::unordered_map<size_t, int> chunk_fds;

  for (auto& group_info : groups_info) {
    size_t current_offset = 0;
    size_t page_size = 4096;
    void* prev_mapped_end = group_info.contiguous_memory;  // 跟踪前一个映射的结束位置

    LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Mapping group " << group_info.group_id
              << " with " << group_info.tensors.size() << " tensors, total_size=" 
              << group_info.total_size << " bytes";

    for (size_t tensor_idx = 0; tensor_idx < group_info.tensors.size(); ++tensor_idx) {
      const auto& tensor_info = group_info.tensors[tensor_idx];
      
      // 确保 current_offset 是页对齐的
      current_offset = ((current_offset + page_size - 1) / page_size) * page_size;
      
      // 确保 current_offset 不会覆盖前一个映射
      size_t prev_mapped_end_offset = static_cast<size_t>(
        static_cast<char*>(prev_mapped_end) - static_cast<char*>(group_info.contiguous_memory)
      );
      if (current_offset < prev_mapped_end_offset) {
        current_offset = prev_mapped_end_offset;
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] WARNING: Group " << group_info.group_id
                  << " tensor " << tensor_info.name
                  << " adjusted current_offset to avoid overlap: " << current_offset;
      }
      
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Group " << group_info.group_id
                << " tensor " << tensor_idx << "/" << group_info.tensors.size()
                << " name=" << tensor_info.name
                << " size=" << tensor_info.size << " bytes"
                << " current_offset=" << current_offset << " (page-aligned)";
      
      int shm_fd;
      if (chunk_fds.find(tensor_info.chunk_id) == chunk_fds.end()) {
        if (tensor_info.chunk_id >= shm_names.size()) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Chunk ID " << tensor_info.chunk_id 
                    << " exceeds shm_names size " << shm_names.size();
          continue;
        }
        const std::string& shm_name = shm_names[tensor_info.chunk_id];
        shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0);
        if (shm_fd == -1) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Failed to open shared memory " 
                    << shm_name << ": " << strerror(errno);
          continue;
        }
        chunk_fds[tensor_info.chunk_id] = shm_fd;
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Opened shared memory " << shm_name 
                  << " (chunk_id=" << tensor_info.chunk_id << ")";
      } else {
        shm_fd = chunk_fds[tensor_info.chunk_id];
      }

      void* target_addr = static_cast<char*>(group_info.contiguous_memory) + current_offset;
      off_t shm_offset = static_cast<off_t>(tensor_info.chunk_offset);
      off_t aligned_shm_offset = (shm_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
      off_t offset_adjustment = shm_offset - aligned_shm_offset;

      // 关键修复：如果 tensor 跨越 chunk 边界，必须使用 memcpy 复制数据
      // 因为 mmap 一次只能映射一个 chunk，无法处理跨 chunk 的情况
      if (tensor_info.crosses_chunk_boundary) {
        // 跨 chunk 的 tensor，使用 memcpy 复制数据
        size_t total_copied = 0;
        size_t remaining_size = tensor_info.size;
        size_t current_chunk_id = tensor_info.chunk_id;
        size_t current_chunk_offset = tensor_info.chunk_offset;
        
        while (remaining_size > 0) {
          // 计算当前 chunk 中可复制的数据量
          size_t chunk_remaining = chunk_size - current_chunk_offset;
          size_t copy_size = (remaining_size < chunk_remaining) ? remaining_size : chunk_remaining;
          
          // 打开当前 chunk 的共享内存
          int shm_fd_current;
          if (chunk_fds.find(current_chunk_id) == chunk_fds.end()) {
            if (current_chunk_id >= shm_names.size()) {
              LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Chunk ID " << current_chunk_id 
                        << " exceeds shm_names size " << shm_names.size();
              break;
            }
            const std::string& shm_name = shm_names[current_chunk_id];
            shm_fd_current = shm_open(shm_name.c_str(), O_RDWR, 0);
            if (shm_fd_current == -1) {
              LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Failed to open shared memory " 
                        << shm_name << ": " << strerror(errno);
              break;
            }
            chunk_fds[current_chunk_id] = shm_fd_current;
          } else {
            shm_fd_current = chunk_fds[current_chunk_id];
          }
          
          // 映射当前 chunk 的数据
          off_t aligned_chunk_offset = (current_chunk_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
          off_t chunk_offset_adjustment = current_chunk_offset - aligned_chunk_offset;
          size_t map_size = ((copy_size + chunk_offset_adjustment + page_size - 1) / page_size) * page_size;
          
          void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd_current, aligned_chunk_offset);
          if (temp_mapped_addr == MAP_FAILED) {
            LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Failed to mmap temp for cross-chunk tensor " 
                      << tensor_info.name << " chunk " << current_chunk_id << ": " << strerror(errno);
            break;
          }
          
          // 复制数据
          void* src_addr = static_cast<char*>(temp_mapped_addr) + chunk_offset_adjustment;
          void* dst_addr = static_cast<char*>(target_addr) + total_copied;
          std::memcpy(dst_addr, src_addr, copy_size);
          
          // 取消映射
          munmap(temp_mapped_addr, map_size);
          
          total_copied += copy_size;
          remaining_size -= copy_size;
          current_chunk_id++;
          current_chunk_offset = 0;
        }
        
        if (total_copied != tensor_info.size) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Failed to copy cross-chunk tensor " 
                    << tensor_info.name << ": copied " << total_copied << " / " << tensor_info.size << " bytes";
          continue;
        }
        
        // 更新 prev_mapped_end 和 current_offset
        prev_mapped_end = static_cast<char*>(target_addr) + tensor_info.size;
        current_offset += tensor_info.size;
        
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Group " << group_info.group_id
                  << " tensor " << tensor_info.name
                  << " copied via memcpy (crosses chunks " << tensor_info.chunk_id 
                  << " -> " << tensor_info.end_chunk_id << ")"
                  << " target_addr=" << target_addr
                  << " size=" << tensor_info.size;
      } else if (offset_adjustment == 0) {
        // 最优情况：offset_adjustment == 0，可以直接映射，无需复制
        size_t map_size = ((tensor_info.size + page_size - 1) / page_size) * page_size;
        void* aligned_target_addr = target_addr;  // 已经是页对齐的
        
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Group " << group_info.group_id
                  << " tensor " << tensor_info.name
                  << " mmap: target_addr=" << target_addr
                  << " map_size=" << map_size
                  << " shm_offset=" << shm_offset << " (page-aligned)";

      void* mapped_addr = mmap(aligned_target_addr, map_size, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_FIXED, shm_fd, aligned_shm_offset);

      if (mapped_addr == MAP_FAILED || mapped_addr != aligned_target_addr) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Failed to mmap for group " 
                    << group_info.group_id << " tensor " << tensor_info.name 
                    << ": " << strerror(errno);
        continue;
      }

        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Successfully mapped group " 
                  << group_info.group_id << " tensor " << tensor_info.name
                  << " at " << mapped_addr << " size=" << map_size;
        
        // 确保数据已同步到内存（避免访问未加载的内存导致 Bus error）
        msync(mapped_addr, map_size, MS_SYNC);
        
        // 更新 prev_mapped_end 和 current_offset
        prev_mapped_end = static_cast<char*>(mapped_addr) + map_size;
      current_offset += tensor_info.size;
      } else {
        // offset_adjustment > 0，使用 memcpy 复制数据，确保完全连续且无重叠
        size_t map_size = ((tensor_info.size + offset_adjustment + page_size - 1) / page_size) * page_size;
        void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd, aligned_shm_offset);
        
        if (temp_mapped_addr == MAP_FAILED) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] ERROR: Failed to mmap temp for group " 
                    << group_info.group_id << " tensor " << tensor_info.name 
                    << ": " << strerror(errno);
          continue;
        }
        
        // 复制数据到目标地址（跳过 offset_adjustment 字节）
        void* src_addr = static_cast<char*>(temp_mapped_addr) + offset_adjustment;
        std::memcpy(target_addr, src_addr, tensor_info.size);
        
        // 取消映射临时地址
        munmap(temp_mapped_addr, map_size);
        
        // 更新 prev_mapped_end 和 current_offset
        prev_mapped_end = static_cast<char*>(target_addr) + tensor_info.size;
        current_offset += tensor_info.size;
        
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Group " << group_info.group_id
                  << " tensor " << tensor_info.name
                  << " copied via memcpy (offset_adjustment=" << offset_adjustment << ")"
                  << " target_addr=" << target_addr
                  << " size=" << tensor_info.size;
      }
      
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Group " << group_info.group_id
                << " tensor " << tensor_info.name
                << " updated current_offset=" << current_offset
                << " prev_mapped_end=" << prev_mapped_end;
      
      // 检查是否超出分配的内存范围
      if (current_offset > group_info.total_size) {
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] WARNING: Group " << group_info.group_id
                  << " current_offset=" << current_offset 
                  << " exceeds total_size=" << group_info.total_size;
      }
    }
    
    LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemory] Group " << group_info.group_id
              << " mapping completed, final_offset=" << current_offset 
              << " total_size=" << group_info.total_size;
  }

  for (auto& [chunk_id, fd] : chunk_fds) {
    close(fd);
  }

  // 第四步：为每个 group 创建 big_tensor
  static std::unordered_map<void*, std::shared_ptr<std::atomic<int>>> memory_ref_counts;
  static std::mutex ref_count_mutex;

  for (auto& group_info : groups_info) {
    torch::Tensor big_tensor = torch::from_blob(
        group_info.contiguous_memory,
        c10::makeArrayRef(group_info.big_tensor_shape),
        c10::makeArrayRef(group_info.big_tensor_strides),
        [](void* ptr) {},
        torch::TensorOptions().device(torch::kCPU).dtype(group_info.dtype));

    std::string big_tensor_key = "group_" + std::to_string(group_info.group_id) + "_big_tensor";

    std::lock_guard<std::mutex> lock(ref_count_mutex);
    auto ref_count = std::make_shared<std::atomic<int>>(1);
    memory_ref_counts[group_info.contiguous_memory] = ref_count;

    torch::Tensor final_big_tensor = torch::from_blob(
        big_tensor.data_ptr(),
        big_tensor.sizes(),
        big_tensor.strides(),
        [ref_count, contiguous_memory = group_info.contiguous_memory, total_size = group_info.total_size](void* ptr) {
          int remaining = ref_count->fetch_sub(1) - 1;
          if (remaining == 0) {
            munmap(contiguous_memory, total_size);
            std::lock_guard<std::mutex> lock(ref_count_mutex);
            memory_ref_counts.erase(contiguous_memory);
          }
        },
        big_tensor.options());

    state_dict[big_tensor_key] = final_big_tensor;

    LOG(INFO) << "Created big_tensor for group " << group_info.group_id
              << " with shape [" << group_info.tensors.size() << ", ...]"
              << " size " << (group_info.total_size / 1024 / 1024) << " MB";
  }

  LOG(INFO) << "Created " << groups_info.size() << " big_tensors from "
            << name_groups.size() << " groups";

  return state_dict;
}

// 性能分析版本：使用 std::cerr 打印各部分耗时，用于找出性能瓶颈
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemoryProfiled(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups
) {
  auto total_start = std::chrono::high_resolution_clock::now();
  std::unordered_map<std::string, torch::Tensor> state_dict;

  if (shm_names.empty()) {
    std::cerr << "[PROFILE] shm_names is empty" << std::endl;
    return {};
  }

  if (name_groups.empty()) {
    std::cerr << "[PROFILE] name_groups is empty" << std::endl;
    return {};
  }

  const size_t page_size = sysconf(_SC_PAGESIZE);

  // Tensor 信息结构
  struct TensorInfo {
    std::string name;
    uint64_t offset;
    uint64_t size;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::string dtype;
    size_t chunk_id;
    size_t chunk_offset;
    size_t group_id;
    bool crosses_chunk_boundary;
    size_t end_chunk_id;
  };

  // Group 信息结构
  struct GroupInfo {
    size_t group_id;
    std::vector<TensorInfo> tensors;
    size_t total_size;
    void* contiguous_memory;
    std::vector<int64_t> big_tensor_shape;
    std::vector<int64_t> big_tensor_strides;
    at::ScalarType dtype;
  };

  std::vector<GroupInfo> groups_info;

  // 第一步：收集每个 group 的 tensor 信息
  auto step1_start = std::chrono::high_resolution_clock::now();
  for (size_t group_id = 0; group_id < name_groups.size(); ++group_id) {
    const auto& name_group = name_groups[group_id];
    
    if (name_group.empty()) {
      continue;
    }

    GroupInfo group_info;
    group_info.group_id = group_id;
    group_info.total_size = 0;
    group_info.contiguous_memory = nullptr;

    for (const std::string& name : name_group) {
      auto it = tensor_metadata.find(name);
      if (it == tensor_metadata.end()) {
        continue;
      }

      auto [offset, tensor_size_bytes, shape, strides, dtype_str] = it->second;

      size_t chunk_id = offset / chunk_size;
      size_t chunk_offset = offset % chunk_size;
      size_t end_chunk_id = (offset + tensor_size_bytes - 1) / chunk_size;
      bool crosses_chunk_boundary = (chunk_id != end_chunk_id);

      TensorInfo info;
      info.name = name;
      info.offset = offset;
      info.size = tensor_size_bytes;
      info.shape = shape;
      info.strides = strides;
      info.dtype = dtype_str;
      info.chunk_id = chunk_id;
      info.chunk_offset = chunk_offset;
      info.group_id = group_id;
      info.crosses_chunk_boundary = crosses_chunk_boundary;
      info.end_chunk_id = end_chunk_id;

      group_info.tensors.push_back(info);
      group_info.total_size += tensor_size_bytes;
    }

    if (group_info.tensors.empty()) {
      continue;
    }

    const auto& first_tensor = group_info.tensors[0];
    bool same_shape = true;
    for (const auto& tensor_info : group_info.tensors) {
      if (tensor_info.shape != first_tensor.shape || tensor_info.dtype != first_tensor.dtype) {
        same_shape = false;
        break;
      }
    }

    if (!same_shape) {
      continue;
    }

    group_info.big_tensor_shape.push_back(static_cast<int64_t>(group_info.tensors.size()));
    for (size_t s : first_tensor.shape) {
      group_info.big_tensor_shape.push_back(static_cast<int64_t>(s));
    }

    int64_t stride = 1;
    for (int i = static_cast<int>(group_info.big_tensor_shape.size()) - 1; i >= 0; --i) {
      group_info.big_tensor_strides.insert(group_info.big_tensor_strides.begin(), stride);
      stride *= group_info.big_tensor_shape[i];
    }

    group_info.dtype = stringToScalarType(first_tensor.dtype);
    group_info.total_size = ((group_info.total_size + page_size - 1) / page_size) * page_size;

    groups_info.push_back(group_info);
  }

  if (groups_info.empty()) {
    std::cerr << "[PROFILE] No valid groups found" << std::endl;
    return {};
  }
  auto step1_end = std::chrono::high_resolution_clock::now();
  auto step1_duration = std::chrono::duration_cast<std::chrono::microseconds>(step1_end - step1_start);
  std::cerr << "[PROFILE] Step1: Collect tensor info cost " 
            << step1_duration.count() / 1000.0 << " ms" << std::endl;

  // 第二步：为每个 group 分配连续的虚拟地址空间
  auto step2_start = std::chrono::high_resolution_clock::now();
  for (auto& group_info : groups_info) {
    void* contiguous_memory = mmap(nullptr, group_info.total_size, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (contiguous_memory == MAP_FAILED) {
      std::cerr << "[PROFILE] ERROR: Failed to allocate contiguous virtual address space for group "
                << group_info.group_id << " of size " << group_info.total_size
                << ": " << strerror(errno) << std::endl;
      for (auto& prev_group : groups_info) {
        if (prev_group.contiguous_memory != nullptr) {
          munmap(prev_group.contiguous_memory, prev_group.total_size);
        }
      }
      return {};
    }
    group_info.contiguous_memory = contiguous_memory;
  }
  auto step2_end = std::chrono::high_resolution_clock::now();
  auto step2_duration = std::chrono::duration_cast<std::chrono::microseconds>(step2_end - step2_start);
  std::cerr << "[PROFILE] Step2: Allocate contiguous memory for " << groups_info.size() 
            << " groups cost " << step2_duration.count() / 1000.0 << " ms" << std::endl;

  // 第三步：为每个 group 映射数据
  auto step3_start = std::chrono::high_resolution_clock::now();
  std::unordered_map<size_t, int> chunk_fds;
  size_t total_shm_open_calls = 0;
  size_t total_mmap_calls = 0;
  size_t total_memcpy_bytes = 0;
  size_t total_cross_chunk_tensors = 0;

  for (auto& group_info : groups_info) {
    size_t current_offset = 0;
    size_t page_size = 4096;
    void* prev_mapped_end = group_info.contiguous_memory;

    auto group_map_start = std::chrono::high_resolution_clock::now();

    for (size_t tensor_idx = 0; tensor_idx < group_info.tensors.size(); ++tensor_idx) {
      const auto& tensor_info = group_info.tensors[tensor_idx];
      
      current_offset = ((current_offset + page_size - 1) / page_size) * page_size;
      
      size_t prev_mapped_end_offset = static_cast<size_t>(
        static_cast<char*>(prev_mapped_end) - static_cast<char*>(group_info.contiguous_memory)
      );
      if (current_offset < prev_mapped_end_offset) {
        current_offset = prev_mapped_end_offset;
      }
      
      int shm_fd;
      if (chunk_fds.find(tensor_info.chunk_id) == chunk_fds.end()) {
        if (tensor_info.chunk_id >= shm_names.size()) {
          continue;
        }
        const std::string& shm_name = shm_names[tensor_info.chunk_id];
        auto shm_open_start = std::chrono::high_resolution_clock::now();
        shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0);
        auto shm_open_end = std::chrono::high_resolution_clock::now();
        auto shm_open_duration = std::chrono::duration_cast<std::chrono::microseconds>(shm_open_end - shm_open_start);
        if (shm_fd == -1) {
          continue;
        }
        chunk_fds[tensor_info.chunk_id] = shm_fd;
        total_shm_open_calls++;
        if (total_shm_open_calls <= 5) {  // 只打印前5次，避免日志过多
          std::cerr << "[PROFILE] shm_open(" << shm_name << ") cost " 
                    << shm_open_duration.count() / 1000.0 << " ms" << std::endl;
        }
      } else {
        shm_fd = chunk_fds[tensor_info.chunk_id];
      }

      void* target_addr = static_cast<char*>(group_info.contiguous_memory) + current_offset;
      off_t shm_offset = static_cast<off_t>(tensor_info.chunk_offset);
      off_t aligned_shm_offset = (shm_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
      off_t offset_adjustment = shm_offset - aligned_shm_offset;

      if (tensor_info.crosses_chunk_boundary) {
        total_cross_chunk_tensors++;
        auto cross_chunk_start = std::chrono::high_resolution_clock::now();
        size_t total_copied = 0;
        size_t remaining_size = tensor_info.size;
        size_t current_chunk_id = tensor_info.chunk_id;
        size_t current_chunk_offset = tensor_info.chunk_offset;
        
        while (remaining_size > 0) {
          size_t chunk_remaining = chunk_size - current_chunk_offset;
          size_t copy_size = (remaining_size < chunk_remaining) ? remaining_size : chunk_remaining;
          
          int shm_fd_current;
          if (chunk_fds.find(current_chunk_id) == chunk_fds.end()) {
            if (current_chunk_id >= shm_names.size()) {
              break;
            }
            const std::string& shm_name = shm_names[current_chunk_id];
            shm_fd_current = shm_open(shm_name.c_str(), O_RDWR, 0);
            if (shm_fd_current == -1) {
              break;
            }
            chunk_fds[current_chunk_id] = shm_fd_current;
            total_shm_open_calls++;
          } else {
            shm_fd_current = chunk_fds[current_chunk_id];
          }
          
          off_t aligned_chunk_offset = (current_chunk_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
          off_t chunk_offset_adjustment = current_chunk_offset - aligned_chunk_offset;
          size_t map_size = ((copy_size + chunk_offset_adjustment + page_size - 1) / page_size) * page_size;
          
          auto mmap_start = std::chrono::high_resolution_clock::now();
          void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd_current, aligned_chunk_offset);
          auto mmap_end = std::chrono::high_resolution_clock::now();
          auto mmap_duration = std::chrono::duration_cast<std::chrono::microseconds>(mmap_end - mmap_start);
          total_mmap_calls++;
          
          if (temp_mapped_addr == MAP_FAILED) {
            break;
          }
          
          void* src_addr = static_cast<char*>(temp_mapped_addr) + chunk_offset_adjustment;
          void* dst_addr = static_cast<char*>(target_addr) + total_copied;
          
          auto memcpy_start = std::chrono::high_resolution_clock::now();
          std::memcpy(dst_addr, src_addr, copy_size);
          auto memcpy_end = std::chrono::high_resolution_clock::now();
          auto memcpy_duration = std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - memcpy_start);
          total_memcpy_bytes += copy_size;
          
          munmap(temp_mapped_addr, map_size);
          
          total_copied += copy_size;
          remaining_size -= copy_size;
          current_chunk_id++;
          current_chunk_offset = 0;
        }
        
        auto cross_chunk_end = std::chrono::high_resolution_clock::now();
        auto cross_chunk_duration = std::chrono::duration_cast<std::chrono::microseconds>(cross_chunk_end - cross_chunk_start);
        if (total_cross_chunk_tensors <= 3) {  // 只打印前3个跨chunk tensor的详细信息
          std::cerr << "[PROFILE] Cross-chunk tensor " << tensor_info.name 
                    << " (size=" << tensor_info.size / 1024 / 1024 << " MB) cost " 
                    << cross_chunk_duration.count() / 1000.0 << " ms" << std::endl;
        }
        
        prev_mapped_end = static_cast<char*>(target_addr) + tensor_info.size;
        current_offset += tensor_info.size;
      } else if (offset_adjustment == 0) {
        size_t map_size = ((tensor_info.size + page_size - 1) / page_size) * page_size;
        void* aligned_target_addr = target_addr;

        auto mmap_start = std::chrono::high_resolution_clock::now();
        void* mapped_addr = mmap(aligned_target_addr, map_size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_FIXED, shm_fd, aligned_shm_offset);
        auto mmap_end = std::chrono::high_resolution_clock::now();
        auto mmap_duration = std::chrono::duration_cast<std::chrono::microseconds>(mmap_end - mmap_start);
        total_mmap_calls++;

        if (mapped_addr == MAP_FAILED || mapped_addr != aligned_target_addr) {
          continue;
        }

        msync(mapped_addr, map_size, MS_SYNC);
        
        prev_mapped_end = static_cast<char*>(mapped_addr) + map_size;
        current_offset += tensor_info.size;
      } else {
        size_t map_size = ((tensor_info.size + offset_adjustment + page_size - 1) / page_size) * page_size;
        
        auto mmap_start = std::chrono::high_resolution_clock::now();
        void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd, aligned_shm_offset);
        auto mmap_end = std::chrono::high_resolution_clock::now();
        auto mmap_duration = std::chrono::duration_cast<std::chrono::microseconds>(mmap_end - mmap_start);
        total_mmap_calls++;
        
        if (temp_mapped_addr == MAP_FAILED) {
          continue;
        }
        
        void* src_addr = static_cast<char*>(temp_mapped_addr) + offset_adjustment;
        
        auto memcpy_start = std::chrono::high_resolution_clock::now();
        std::memcpy(target_addr, src_addr, tensor_info.size);
        auto memcpy_end = std::chrono::high_resolution_clock::now();
        auto memcpy_duration = std::chrono::duration_cast<std::chrono::microseconds>(memcpy_end - memcpy_start);
        total_memcpy_bytes += tensor_info.size;
        
        munmap(temp_mapped_addr, map_size);
        
        prev_mapped_end = static_cast<char*>(target_addr) + tensor_info.size;
        current_offset += tensor_info.size;
      }
    }
    
    auto group_map_end = std::chrono::high_resolution_clock::now();
    auto group_map_duration = std::chrono::duration_cast<std::chrono::microseconds>(group_map_end - group_map_start);
    std::cerr << "[PROFILE] Group " << group_info.group_id << " mapping cost " 
              << group_map_duration.count() / 1000.0 << " ms" 
              << " (tensors=" << group_info.tensors.size() 
              << ", size=" << group_info.total_size / 1024 / 1024 << " MB)" << std::endl;
  }

  for (auto& [chunk_id, fd] : chunk_fds) {
    close(fd);
  }
  auto step3_end = std::chrono::high_resolution_clock::now();
  auto step3_duration = std::chrono::duration_cast<std::chrono::microseconds>(step3_end - step3_start);
  std::cerr << "[PROFILE] Step3: Map shared memory cost " 
            << step3_duration.count() / 1000.0 << " ms" << std::endl;
  std::cerr << "[PROFILE] Step3 details: shm_open calls=" << total_shm_open_calls 
            << ", mmap calls=" << total_mmap_calls 
            << ", memcpy bytes=" << total_memcpy_bytes / 1024 / 1024 << " MB"
            << ", cross-chunk tensors=" << total_cross_chunk_tensors << std::endl;

  // 第四步：为每个 group 创建 big_tensor
  auto step4_start = std::chrono::high_resolution_clock::now();
  static std::unordered_map<void*, std::shared_ptr<std::atomic<int>>> memory_ref_counts;
  static std::mutex ref_count_mutex;

  for (auto& group_info : groups_info) {
    torch::Tensor big_tensor = torch::from_blob(
        group_info.contiguous_memory,
        c10::makeArrayRef(group_info.big_tensor_shape),
        c10::makeArrayRef(group_info.big_tensor_strides),
        [](void* ptr) {},
        torch::TensorOptions().device(torch::kCPU).dtype(group_info.dtype));

    // std::string big_tensor_key = "group_" + std::to_string(group_info.group_id) + "_big_tensor";
    std::string big_tensor_key = "group_w" + std::to_string(group_info.group_id+1); // w1, w2, w3

    std::lock_guard<std::mutex> lock(ref_count_mutex);
    auto ref_count = std::make_shared<std::atomic<int>>(1);
    memory_ref_counts[group_info.contiguous_memory] = ref_count;

    torch::Tensor final_big_tensor = torch::from_blob(
        big_tensor.data_ptr(),
        big_tensor.sizes(),
        big_tensor.strides(),
        [ref_count, contiguous_memory = group_info.contiguous_memory, total_size = group_info.total_size](void* ptr) {
          int remaining = ref_count->fetch_sub(1) - 1;
          if (remaining == 0) {
            munmap(contiguous_memory, total_size);
            std::lock_guard<std::mutex> lock(ref_count_mutex);
            memory_ref_counts.erase(contiguous_memory);
          }
        },
        big_tensor.options());

    state_dict[big_tensor_key] = final_big_tensor;
  }
  auto step4_end = std::chrono::high_resolution_clock::now();
  auto step4_duration = std::chrono::duration_cast<std::chrono::microseconds>(step4_end - step4_start);
  std::cerr << "[PROFILE] Step4: Create tensors cost " 
            << step4_duration.count() / 1000.0 << " ms" << std::endl;

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start);
  std::cerr << "[PROFILE] ========================================" << std::endl;
  std::cerr << "[PROFILE] Total cost " << total_duration.count() / 1000.0 << " ms" << std::endl;
  std::cerr << "[PROFILE] Breakdown:" << std::endl;
  if (total_duration.count() > 0) {
    std::cerr << "[PROFILE]   Step1 (Collect info): " << step1_duration.count() / 1000.0 << " ms (" 
              << (step1_duration.count() * 100.0 / total_duration.count()) << "%)" << std::endl;
    std::cerr << "[PROFILE]   Step2 (Allocate memory): " << step2_duration.count() / 1000.0 << " ms (" 
              << (step2_duration.count() * 100.0 / total_duration.count()) << "%)" << std::endl;
    std::cerr << "[PROFILE]   Step3 (Map memory): " << step3_duration.count() / 1000.0 << " ms (" 
              << (step3_duration.count() * 100.0 / total_duration.count()) << "%)" << std::endl;
    std::cerr << "[PROFILE]   Step4 (Create tensors): " << step4_duration.count() / 1000.0 << " ms (" 
              << (step4_duration.count() * 100.0 / total_duration.count()) << "%)" << std::endl;
  } else {
    std::cerr << "[PROFILE]   Step1 (Collect info): " << step1_duration.count() / 1000.0 << " ms" << std::endl;
    std::cerr << "[PROFILE]   Step2 (Allocate memory): " << step2_duration.count() / 1000.0 << " ms" << std::endl;
    std::cerr << "[PROFILE]   Step3 (Map memory): " << step3_duration.count() / 1000.0 << " ms" << std::endl;
    std::cerr << "[PROFILE]   Step4 (Create tensors): " << step4_duration.count() / 1000.0 << " ms" << std::endl;
  }
  std::cerr << "[PROFILE] ========================================" << std::endl;

  return state_dict;
}

// 静默版本：功能与 RestoreExpertsGroupsFromSharedMemoryProfiled 相同，但不输出任何日志
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemorySilent(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups
) {
  std::unordered_map<std::string, torch::Tensor> state_dict;

  if (shm_names.empty() || name_groups.empty()) {
    return {};
  }

  const size_t page_size = sysconf(_SC_PAGESIZE);

  // Tensor 信息结构
  struct TensorInfo {
    std::string name;
    uint64_t offset;
    uint64_t size;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::string dtype;
    size_t chunk_id;
    size_t chunk_offset;
    size_t group_id;
    bool crosses_chunk_boundary;
    size_t end_chunk_id;
  };

  // Group 信息结构
  struct GroupInfo {
    size_t group_id;
    std::vector<TensorInfo> tensors;
    size_t total_size;
    void* contiguous_memory;
    std::vector<int64_t> big_tensor_shape;
    std::vector<int64_t> big_tensor_strides;
    at::ScalarType dtype;
  };

  std::vector<GroupInfo> groups_info;

  // 第一步：收集每个 group 的 tensor 信息
  for (size_t group_id = 0; group_id < name_groups.size(); ++group_id) {
    const auto& name_group = name_groups[group_id];
    
    if (name_group.empty()) {
      continue;
    }

    GroupInfo group_info;
    group_info.group_id = group_id;
    group_info.total_size = 0;
    group_info.contiguous_memory = nullptr;

    for (const std::string& name : name_group) {
      auto it = tensor_metadata.find(name);
      if (it == tensor_metadata.end()) {
        continue;
      }

      auto [offset, tensor_size_bytes, shape, strides, dtype_str] = it->second;

      size_t chunk_id = offset / chunk_size;
      size_t chunk_offset = offset % chunk_size;
      size_t end_chunk_id = (offset + tensor_size_bytes - 1) / chunk_size;
      bool crosses_chunk_boundary = (chunk_id != end_chunk_id);

      TensorInfo info;
      info.name = name;
      info.offset = offset;
      info.size = tensor_size_bytes;
      info.shape = shape;
      info.strides = strides;
      info.dtype = dtype_str;
      info.chunk_id = chunk_id;
      info.chunk_offset = chunk_offset;
      info.group_id = group_id;
      info.crosses_chunk_boundary = crosses_chunk_boundary;
      info.end_chunk_id = end_chunk_id;

      group_info.tensors.push_back(info);
      group_info.total_size += tensor_size_bytes;
    }

    if (group_info.tensors.empty()) {
      continue;
    }

    const auto& first_tensor = group_info.tensors[0];
    bool same_shape = true;
    for (const auto& tensor_info : group_info.tensors) {
      if (tensor_info.shape != first_tensor.shape || tensor_info.dtype != first_tensor.dtype) {
        same_shape = false;
        break;
      }
    }

    if (!same_shape) {
      continue;
    }

    group_info.big_tensor_shape.push_back(static_cast<int64_t>(group_info.tensors.size()));
    for (size_t s : first_tensor.shape) {
      group_info.big_tensor_shape.push_back(static_cast<int64_t>(s));
    }

    int64_t stride = 1;
    for (int i = static_cast<int>(group_info.big_tensor_shape.size()) - 1; i >= 0; --i) {
      group_info.big_tensor_strides.insert(group_info.big_tensor_strides.begin(), stride);
      stride *= group_info.big_tensor_shape[i];
    }

    group_info.dtype = stringToScalarType(first_tensor.dtype);
    group_info.total_size = ((group_info.total_size + page_size - 1) / page_size) * page_size;

    groups_info.push_back(group_info);
  }

  if (groups_info.empty()) {
    return {};
  }

  // 第二步：为每个 group 分配连续的虚拟地址空间
  for (auto& group_info : groups_info) {
    void* contiguous_memory = mmap(nullptr, group_info.total_size, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (contiguous_memory == MAP_FAILED) {
      for (auto& prev_group : groups_info) {
        if (prev_group.contiguous_memory != nullptr) {
          munmap(prev_group.contiguous_memory, prev_group.total_size);
        }
      }
      return {};
    }
    group_info.contiguous_memory = contiguous_memory;
  }

  // 第三步：为每个 group 映射数据
  std::unordered_map<size_t, int> chunk_fds;

  for (auto& group_info : groups_info) {
    size_t current_offset = 0;
    size_t page_size = 4096;
    void* prev_mapped_end = group_info.contiguous_memory;

    for (size_t tensor_idx = 0; tensor_idx < group_info.tensors.size(); ++tensor_idx) {
      const auto& tensor_info = group_info.tensors[tensor_idx];
      
      current_offset = ((current_offset + page_size - 1) / page_size) * page_size;
      
      size_t prev_mapped_end_offset = static_cast<size_t>(
        static_cast<char*>(prev_mapped_end) - static_cast<char*>(group_info.contiguous_memory)
      );
      if (current_offset < prev_mapped_end_offset) {
        current_offset = prev_mapped_end_offset;
      }
      
      int shm_fd;
      if (chunk_fds.find(tensor_info.chunk_id) == chunk_fds.end()) {
        if (tensor_info.chunk_id >= shm_names.size()) {
          continue;
        }
        const std::string& shm_name = shm_names[tensor_info.chunk_id];
        shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0);
        if (shm_fd == -1) {
          continue;
        }
        chunk_fds[tensor_info.chunk_id] = shm_fd;
      } else {
        shm_fd = chunk_fds[tensor_info.chunk_id];
      }

      void* target_addr = static_cast<char*>(group_info.contiguous_memory) + current_offset;
      off_t shm_offset = static_cast<off_t>(tensor_info.chunk_offset);
      off_t aligned_shm_offset = (shm_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
      off_t offset_adjustment = shm_offset - aligned_shm_offset;

      if (tensor_info.crosses_chunk_boundary) {
        size_t total_copied = 0;
        size_t remaining_size = tensor_info.size;
        size_t current_chunk_id = tensor_info.chunk_id;
        size_t current_chunk_offset = tensor_info.chunk_offset;
        
        while (remaining_size > 0) {
          size_t chunk_remaining = chunk_size - current_chunk_offset;
          size_t copy_size = (remaining_size < chunk_remaining) ? remaining_size : chunk_remaining;
          
          int shm_fd_current;
          if (chunk_fds.find(current_chunk_id) == chunk_fds.end()) {
            if (current_chunk_id >= shm_names.size()) {
              break;
            }
            const std::string& shm_name = shm_names[current_chunk_id];
            shm_fd_current = shm_open(shm_name.c_str(), O_RDWR, 0);
            if (shm_fd_current == -1) {
              break;
            }
            chunk_fds[current_chunk_id] = shm_fd_current;
          } else {
            shm_fd_current = chunk_fds[current_chunk_id];
          }
          
          off_t aligned_chunk_offset = (current_chunk_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
          off_t chunk_offset_adjustment = current_chunk_offset - aligned_chunk_offset;
          size_t map_size = ((copy_size + chunk_offset_adjustment + page_size - 1) / page_size) * page_size;
          
          void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd_current, aligned_chunk_offset);
          
          if (temp_mapped_addr == MAP_FAILED) {
            break;
          }
          
          void* src_addr = static_cast<char*>(temp_mapped_addr) + chunk_offset_adjustment;
          void* dst_addr = static_cast<char*>(target_addr) + total_copied;
          
          std::memcpy(dst_addr, src_addr, copy_size);
          
          munmap(temp_mapped_addr, map_size);
          
          total_copied += copy_size;
          remaining_size -= copy_size;
          current_chunk_id++;
          current_chunk_offset = 0;
        }
        
        prev_mapped_end = static_cast<char*>(target_addr) + tensor_info.size;
        current_offset += tensor_info.size;
      } else if (offset_adjustment == 0) {
        size_t map_size = ((tensor_info.size + page_size - 1) / page_size) * page_size;
        void* aligned_target_addr = target_addr;

        void* mapped_addr = mmap(aligned_target_addr, map_size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_FIXED, shm_fd, aligned_shm_offset);

        if (mapped_addr == MAP_FAILED || mapped_addr != aligned_target_addr) {
          continue;
        }

        msync(mapped_addr, map_size, MS_SYNC);
        
        prev_mapped_end = static_cast<char*>(mapped_addr) + map_size;
        current_offset += tensor_info.size;
      } else {
        size_t map_size = ((tensor_info.size + offset_adjustment + page_size - 1) / page_size) * page_size;
        
        void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd, aligned_shm_offset);
        
        if (temp_mapped_addr == MAP_FAILED) {
          continue;
        }
        
        void* src_addr = static_cast<char*>(temp_mapped_addr) + offset_adjustment;
        
        std::memcpy(target_addr, src_addr, tensor_info.size);
        
        munmap(temp_mapped_addr, map_size);
        
        prev_mapped_end = static_cast<char*>(target_addr) + tensor_info.size;
        current_offset += tensor_info.size;
      }
    }
  }

  for (auto& [chunk_id, fd] : chunk_fds) {
    close(fd);
  }

  // 第四步：为每个 group 创建 big_tensor
  static std::unordered_map<void*, std::shared_ptr<std::atomic<int>>> memory_ref_counts;
  static std::mutex ref_count_mutex;

  for (auto& group_info : groups_info) {
    torch::Tensor big_tensor = torch::from_blob(
        group_info.contiguous_memory,
        c10::makeArrayRef(group_info.big_tensor_shape),
        c10::makeArrayRef(group_info.big_tensor_strides),
        [](void* ptr) {},
        torch::TensorOptions().device(torch::kCPU).dtype(group_info.dtype));

    std::string big_tensor_key = "group_w" + std::to_string(group_info.group_id+1); // w1, w2, w3

    std::lock_guard<std::mutex> lock(ref_count_mutex);
    auto ref_count = std::make_shared<std::atomic<int>>(1);
    memory_ref_counts[group_info.contiguous_memory] = ref_count;

    torch::Tensor final_big_tensor = torch::from_blob(
        big_tensor.data_ptr(),
        big_tensor.sizes(),
        big_tensor.strides(),
        [ref_count, contiguous_memory = group_info.contiguous_memory, total_size = group_info.total_size](void* ptr) {
          int remaining = ref_count->fetch_sub(1) - 1;
          if (remaining == 0) {
            munmap(contiguous_memory, total_size);
            std::lock_guard<std::mutex> lock(ref_count_mutex);
            memory_ref_counts.erase(contiguous_memory);
          }
        },
        big_tensor.options());

    state_dict[big_tensor_key] = final_big_tensor;
  }

  return state_dict;
}

// 使用缓存的版本，避免每次调用都转换 tensor_metadata
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemoryProfiledCached(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMapCache& tensor_metadata_cache,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups
) {
  // 直接使用缓存的 metadata，避免重复转换
  return RestoreExpertsGroupsFromSharedMemoryProfiled(
      shm_names, tensor_metadata_cache.get(), chunk_size, name_groups);
}

// 静默版本的缓存版本，避免每次调用都转换 tensor_metadata
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemorySilentCached(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMapCache& tensor_metadata_cache,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups
) {
  // 直接使用缓存的 metadata，避免重复转换
  return RestoreExpertsGroupsFromSharedMemorySilent(
      shm_names, tensor_metadata_cache.get(), chunk_size, name_groups);
}

// 全局缓存：存储已分配的虚拟地址空间（不存储数据，只存储地址空间）
// key 为 group 的唯一标识（基于 tensor 名称）
struct CachedGroupMemory {
  void* contiguous_memory;  // 虚拟地址空间
  size_t total_size;         // 分配的虚拟地址空间大小（>= 实际需要的 group_size）
  std::shared_ptr<std::atomic<int>> ref_count;
};

static std::unordered_map<std::string, CachedGroupMemory> group_memory_cache;
static std::unordered_map<void*, std::shared_ptr<std::atomic<int>>> cached_memory_ref_counts;
static std::mutex cache_mutex;

// 支持多个 vector，为每个 vector 创建一个 big_tensor（带缓存版本）
// 这个版本会复用已映射的内存，避免频繁的 mmap/munmap 操作
std::unordered_map<std::string, torch::Tensor> RestoreExpertsGroupsFromSharedMemoryCached(
    const std::vector<std::string>& shm_names,
    const TensorIndexResizeMap& tensor_metadata,
    size_t chunk_size,
    const std::vector<std::vector<std::string>>& name_groups
) {
  std::unordered_map<std::string, torch::Tensor> state_dict;

  if (shm_names.empty()) {
    LOG(INFO) << "shm_names is empty";
    return {};
  }

  if (name_groups.empty()) {
    LOG(INFO) << "name_groups is empty";
    return {};
  }

  const size_t page_size = sysconf(_SC_PAGESIZE);

  // Tensor 信息结构
  struct TensorInfo {
    std::string name;
    uint64_t offset;
    uint64_t size;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    std::string dtype;
    size_t chunk_id;
    size_t chunk_offset;
    size_t group_id;
    bool crosses_chunk_boundary;
    size_t end_chunk_id;
  };

  // Group 信息结构
  struct GroupInfo {
    size_t group_id;
    std::vector<TensorInfo> tensors;
    size_t total_size;
    void* contiguous_memory;
    std::vector<int64_t> big_tensor_shape;
    std::vector<int64_t> big_tensor_strides;
    at::ScalarType dtype;
  };

  std::vector<GroupInfo> groups_info;

  // 第一步：收集每个 group 的 tensor 信息
  for (size_t group_id = 0; group_id < name_groups.size(); ++group_id) {
    const auto& name_group = name_groups[group_id];
    
    if (name_group.empty()) {
      LOG(INFO) << "Group " << group_id << " is empty, skipping";
      continue;
    }

    GroupInfo group_info;
    group_info.group_id = group_id;
    group_info.total_size = 0;
    group_info.contiguous_memory = nullptr;

    for (const std::string& name : name_group) {
      auto it = tensor_metadata.find(name);
      if (it == tensor_metadata.end()) {
        LOG(INFO) << "Tensor " << name << " not found in tensor_metadata";
        continue;
      }

      auto [offset, tensor_size_bytes, shape, strides, dtype_str] = it->second;

      size_t chunk_id = offset / chunk_size;
      size_t chunk_offset = offset % chunk_size;
      size_t end_chunk_id = (offset + tensor_size_bytes - 1) / chunk_size;
      bool crosses_chunk_boundary = (chunk_id != end_chunk_id);

      TensorInfo info;
      info.name = name;
      info.offset = offset;
      info.size = tensor_size_bytes;
      info.shape = shape;
      info.strides = strides;
      info.dtype = dtype_str;
      info.chunk_id = chunk_id;
      info.chunk_offset = chunk_offset;
      info.group_id = group_id;
      info.crosses_chunk_boundary = crosses_chunk_boundary;
      info.end_chunk_id = end_chunk_id;

      group_info.tensors.push_back(info);
      group_info.total_size += tensor_size_bytes;
    }

    if (group_info.tensors.empty()) {
      LOG(INFO) << "Group " << group_id << " has no valid tensors, skipping";
      continue;
    }

    const auto& first_tensor = group_info.tensors[0];
    bool same_shape = true;
    for (const auto& tensor_info : group_info.tensors) {
      if (tensor_info.shape != first_tensor.shape || tensor_info.dtype != first_tensor.dtype) {
        same_shape = false;
        break;
      }
    }

    if (!same_shape) {
      LOG(INFO) << "Group " << group_id << " has tensors with different shapes or dtypes, skipping";
      continue;
    }

    group_info.big_tensor_shape.push_back(static_cast<int64_t>(group_info.tensors.size()));
    for (size_t s : first_tensor.shape) {
      group_info.big_tensor_shape.push_back(static_cast<int64_t>(s));
    }

    int64_t stride = 1;
    for (int i = static_cast<int>(group_info.big_tensor_shape.size()) - 1; i >= 0; --i) {
      group_info.big_tensor_strides.insert(group_info.big_tensor_strides.begin(), stride);
      stride *= group_info.big_tensor_shape[i];
    }

    group_info.dtype = stringToScalarType(first_tensor.dtype);
    group_info.total_size = ((group_info.total_size + page_size - 1) / page_size) * page_size;

    groups_info.push_back(group_info);
  }

  if (groups_info.empty()) {
    LOG(INFO) << "No valid groups found";
    return {};
  }

  // 第二步：检查缓存并复用或分配新的连续虚拟地址空间
  // 不根据名字复用，只要空间足够就复用
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    for (auto& group_info : groups_info) {
      // 计算实际需要的 group_size（未页对齐的原始大小）
      size_t required_size = 0;
      for (const auto& tensor_info : group_info.tensors) {
        required_size += tensor_info.size;
      }
      size_t required_size_aligned = ((required_size + page_size - 1) / page_size) * page_size;
      
      // 遍历所有缓存的虚拟地址空间，找到第一个大小足够且未被使用的就复用
      bool reuse_virtual_address = false;
      void* reused_memory = nullptr;
      size_t reused_size = 0;
      std::string reused_key;
      
      for (auto& [cached_key, cached] : group_memory_cache) {
        // 检查引用计数，只有未被使用的（引用计数为0）才能复用
        int ref_count = cached.ref_count->load();
        if (ref_count > 0) {
          // 正在使用中，跳过
          continue;
        }
        
        // 如果缓存的虚拟地址空间大小足够且未被使用，复用
        if (cached.total_size >= required_size_aligned) {
          group_info.contiguous_memory = cached.contiguous_memory;
          cached.ref_count->fetch_add(1);
          reuse_virtual_address = true;
          reused_memory = cached.contiguous_memory;
          reused_size = cached.total_size;
          reused_key = cached_key;
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Reusing cached virtual address space for group " 
                    << group_info.group_id 
                    << " (cached_size=" << cached.total_size 
                    << ", required_size=" << required_size_aligned 
                    << ", key: " << cached_key 
                    << ", previous_ref_count=" << ref_count << ")";
          break;
        }
      }
      
      // 如果未复用缓存，分配新的虚拟地址空间
      if (!reuse_virtual_address) {
        void* contiguous_memory = mmap(nullptr, required_size_aligned, PROT_READ | PROT_WRITE,
                                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (contiguous_memory == MAP_FAILED) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Failed to allocate contiguous virtual address space for group "
                    << group_info.group_id << " of size " << required_size_aligned
                    << ": " << strerror(errno);
          continue;
        }
        group_info.contiguous_memory = contiguous_memory;
        
        // 生成一个基于地址的 key 用于缓存管理
        std::string address_key = "addr_" + std::to_string(reinterpret_cast<uintptr_t>(contiguous_memory));
        
        // 将新分配的虚拟地址空间添加到缓存
        CachedGroupMemory cached;
        cached.contiguous_memory = group_info.contiguous_memory;
        cached.total_size = required_size_aligned;
        cached.ref_count = std::make_shared<std::atomic<int>>(1);
        group_memory_cache[address_key] = cached;
        cached_memory_ref_counts[group_info.contiguous_memory] = cached.ref_count;
        
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Allocated and cached virtual address space for group " 
                  << group_info.group_id << " at " << contiguous_memory 
                  << " size=" << required_size_aligned << " bytes (key: " << address_key << ")";
      }
      
      // 更新 group_info.total_size 为实际分配的虚拟地址空间大小
      if (reuse_virtual_address) {
        group_info.total_size = reused_size;
      } else {
        group_info.total_size = required_size_aligned;
      }
    }
  }

  // 第三步：为每个 group 映射数据（每次调用都重新映射，因为数据可能变化）
  std::unordered_map<size_t, int> chunk_fds;

  for (auto& group_info : groups_info) {
    // 每次调用都重新映射数据，复用虚拟地址空间但更新映射的数据
    
    size_t current_offset = 0;
    size_t page_size = 4096;
    void* prev_mapped_end = group_info.contiguous_memory;

    LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Mapping group " << group_info.group_id
              << " with " << group_info.tensors.size() << " tensors, total_size=" 
              << group_info.total_size << " bytes"
              << " contiguous_memory=" << group_info.contiguous_memory;

    for (size_t tensor_idx = 0; tensor_idx < group_info.tensors.size(); ++tensor_idx) {
      const auto& tensor_info = group_info.tensors[tensor_idx];
      
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                << " tensor " << tensor_idx << "/" << group_info.tensors.size()
                << " name=" << tensor_info.name << " size=" << tensor_info.size << " bytes";
      
      current_offset = ((current_offset + page_size - 1) / page_size) * page_size;
      
      size_t prev_mapped_end_offset = static_cast<size_t>(
        static_cast<char*>(prev_mapped_end) - static_cast<char*>(group_info.contiguous_memory)
      );
      if (current_offset < prev_mapped_end_offset) {
        current_offset = prev_mapped_end_offset;
      }
      
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                << " tensor " << tensor_idx << " current_offset=" << current_offset;
      
      int shm_fd;
      if (chunk_fds.find(tensor_info.chunk_id) == chunk_fds.end()) {
        if (tensor_info.chunk_id >= shm_names.size()) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Chunk ID " << tensor_info.chunk_id 
                    << " exceeds shm_names size " << shm_names.size();
          continue;
        }
        const std::string& shm_name = shm_names[tensor_info.chunk_id];
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Opening shared memory " 
                  << shm_name << " (chunk_id=" << tensor_info.chunk_id << ")";
        shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0);
        if (shm_fd == -1) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Failed to open shared memory " 
                    << shm_name << ": " << strerror(errno);
          continue;
        }
        chunk_fds[tensor_info.chunk_id] = shm_fd;
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Successfully opened shared memory " 
                  << shm_name << " fd=" << shm_fd;
      } else {
        shm_fd = chunk_fds[tensor_info.chunk_id];
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Reusing fd=" << shm_fd 
                  << " for chunk_id=" << tensor_info.chunk_id;
      }

      void* target_addr = static_cast<char*>(group_info.contiguous_memory) + current_offset;
      off_t shm_offset = static_cast<off_t>(tensor_info.chunk_offset);
      off_t aligned_shm_offset = (shm_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
      off_t offset_adjustment = shm_offset - aligned_shm_offset;
      
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                << " tensor " << tensor_idx << " target_addr=" << target_addr
                << " shm_offset=" << shm_offset << " aligned_shm_offset=" << aligned_shm_offset
                << " offset_adjustment=" << offset_adjustment
                << " crosses_chunk=" << tensor_info.crosses_chunk_boundary;

      if (tensor_info.crosses_chunk_boundary) {
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                  << " tensor " << tensor_idx << " crosses chunk boundary, using memcpy";
        size_t total_copied = 0;
        size_t remaining_size = tensor_info.size;
        size_t current_chunk_id = tensor_info.chunk_id;
        size_t current_chunk_offset = tensor_info.chunk_offset;
        
        while (remaining_size > 0) {
          size_t chunk_remaining = chunk_size - current_chunk_offset;
          size_t copy_size = (remaining_size < chunk_remaining) ? remaining_size : chunk_remaining;
          
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                    << " tensor " << tensor_idx << " copying from chunk " << current_chunk_id
                    << " offset " << current_chunk_offset << " size " << copy_size;
          
          int shm_fd_current;
          if (chunk_fds.find(current_chunk_id) == chunk_fds.end()) {
            if (current_chunk_id >= shm_names.size()) {
              LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Chunk ID " << current_chunk_id 
                        << " exceeds shm_names size";
              break;
            }
            const std::string& shm_name = shm_names[current_chunk_id];
            LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Opening chunk " << current_chunk_id
                      << " shared memory " << shm_name;
            shm_fd_current = shm_open(shm_name.c_str(), O_RDWR, 0);
            if (shm_fd_current == -1) {
              LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Failed to open chunk " 
                        << current_chunk_id << " shared memory: " << strerror(errno);
              break;
            }
            chunk_fds[current_chunk_id] = shm_fd_current;
          } else {
            shm_fd_current = chunk_fds[current_chunk_id];
          }
          
          off_t aligned_chunk_offset = (current_chunk_offset / static_cast<off_t>(page_size)) * static_cast<off_t>(page_size);
          off_t chunk_offset_adjustment = current_chunk_offset - aligned_chunk_offset;
          size_t map_size = ((copy_size + chunk_offset_adjustment + page_size - 1) / page_size) * page_size;
          
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                    << " tensor " << tensor_idx << " mmap temp size=" << map_size 
                    << " offset=" << aligned_chunk_offset;
          
          void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd_current, aligned_chunk_offset);
          if (temp_mapped_addr == MAP_FAILED) {
            LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: mmap failed for chunk " 
                      << current_chunk_id << ": " << strerror(errno);
            break;
          }
          
          void* src_addr = static_cast<char*>(temp_mapped_addr) + chunk_offset_adjustment;
          void* dst_addr = static_cast<char*>(target_addr) + total_copied;
          
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                    << " tensor " << tensor_idx << " memcpy " << copy_size << " bytes";
          std::memcpy(dst_addr, src_addr, copy_size);
          
          munmap(temp_mapped_addr, map_size);
          
          total_copied += copy_size;
          remaining_size -= copy_size;
          current_chunk_id++;
          current_chunk_offset = 0;
        }
        
        if (total_copied != tensor_info.size) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Group " << group_info.group_id
                    << " tensor " << tensor_idx << " copied " << total_copied 
                    << " / " << tensor_info.size << " bytes";
          continue;
        }
        
        prev_mapped_end = static_cast<char*>(target_addr) + tensor_info.size;
        current_offset += tensor_info.size;
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                  << " tensor " << tensor_idx << " cross-chunk copy completed";
      } else if (offset_adjustment == 0) {
        size_t map_size = ((tensor_info.size + page_size - 1) / page_size) * page_size;
        void* aligned_target_addr = target_addr;

        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                  << " tensor " << tensor_idx << " direct mmap size=" << map_size
                  << " target=" << aligned_target_addr << " shm_offset=" << aligned_shm_offset;

        // 使用 MAP_FIXED 可以直接替换现有映射，不需要先 munmap
        // MAP_FIXED 会自动取消目标地址范围内的现有映射并创建新映射
        // 这样可以避免 munmap 的系统调用开销
        void* mapped_addr = mmap(aligned_target_addr, map_size, PROT_READ | PROT_WRITE,
                                 MAP_SHARED | MAP_FIXED, shm_fd, aligned_shm_offset);

        if (mapped_addr == MAP_FAILED || mapped_addr != aligned_target_addr) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Group " << group_info.group_id
                    << " tensor " << tensor_idx << " mmap failed: " << strerror(errno)
                    << " mapped_addr=" << mapped_addr << " expected=" << aligned_target_addr;
          continue;
        }

        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                  << " tensor " << tensor_idx << " mmap successful";
        
        // msync 可能会阻塞很长时间，特别是对于大文件
        // 由于数据已经在共享内存中，msync 可能不是必需的
        // 使用 MS_ASYNC 而不是 MS_SYNC 可以避免阻塞
        // 或者完全移除 msync，因为共享内存的数据应该已经是同步的
        // 暂时注释掉 msync，如果后续发现数据不一致问题再启用
        /*
            LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                      << " tensor " << tensor_idx << " calling msync (async)...";
        int msync_result = msync(mapped_addr, map_size, MS_ASYNC);
        if (msync_result != 0) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] WARNING: Group " << group_info.group_id
                    << " tensor " << tensor_idx << " msync failed: " << strerror(errno);
        }
        */
        
        prev_mapped_end = static_cast<char*>(mapped_addr) + map_size;
        current_offset += tensor_info.size;
      } else {
        size_t map_size = ((tensor_info.size + offset_adjustment + page_size - 1) / page_size) * page_size;
        
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                  << " tensor " << tensor_idx << " memcpy path: map_size=" << map_size
                  << " offset_adjustment=" << offset_adjustment;
        
        void* temp_mapped_addr = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, shm_fd, aligned_shm_offset);
        
        if (temp_mapped_addr == MAP_FAILED) {
          LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Group " << group_info.group_id
                    << " tensor " << tensor_idx << " temp mmap failed: " << strerror(errno);
          continue;
        }
        
        void* src_addr = static_cast<char*>(temp_mapped_addr) + offset_adjustment;
        
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                  << " tensor " << tensor_idx << " memcpy " << tensor_info.size << " bytes";
        std::memcpy(target_addr, src_addr, tensor_info.size);
        
        munmap(temp_mapped_addr, map_size);
        
        prev_mapped_end = static_cast<char*>(target_addr) + tensor_info.size;
        current_offset += tensor_info.size;
        
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                  << " tensor " << tensor_idx << " memcpy completed";
      }
      
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Group " << group_info.group_id
                << " tensor " << tensor_idx << " mapping completed";
    }
  }

  for (auto& [chunk_id, fd] : chunk_fds) {
    close(fd);
  }

  // 第四步：为每个 group 创建 big_tensor
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    for (auto& group_info : groups_info) {
      LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Creating big_tensor for group " 
                << group_info.group_id;
      
      torch::Tensor big_tensor = torch::from_blob(
          group_info.contiguous_memory,
          c10::makeArrayRef(group_info.big_tensor_shape),
          c10::makeArrayRef(group_info.big_tensor_strides),
          [](void* ptr) {},
          torch::TensorOptions().device(torch::kCPU).dtype(group_info.dtype));

      std::string big_tensor_key = "group_" + std::to_string(group_info.group_id) + "_big_tensor";

      // 获取引用计数（从缓存，通过地址查找）
      // 注意：已经在 cache_mutex 锁保护下，不需要再次加锁
      std::shared_ptr<std::atomic<int>> ref_count = nullptr;
      if (cached_memory_ref_counts.find(group_info.contiguous_memory) != cached_memory_ref_counts.end()) {
        ref_count = cached_memory_ref_counts[group_info.contiguous_memory];
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Found ref_count for group " 
                  << group_info.group_id << " at " << group_info.contiguous_memory;
      } else {
        LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] ERROR: Failed to find ref_count for group " 
                  << group_info.group_id << " at " << group_info.contiguous_memory;
        continue;
      }

      // 修改 deleter：不立即 munmap，只减少引用计数
      torch::Tensor final_big_tensor = torch::from_blob(
          big_tensor.data_ptr(),
          big_tensor.sizes(),
          big_tensor.strides(),
          [ref_count, contiguous_memory = group_info.contiguous_memory](void* ptr) {
            // 只减少引用计数，不立即释放内存
            int remaining = ref_count->fetch_sub(1) - 1;
            LOG(INFO) << "[RestoreExpertsGroupsFromSharedMemoryCached] Tensor released, remaining refs: " 
                      << remaining << " for address: " << contiguous_memory;
            // 注意：内存不会在这里释放，需要通过主动调用释放函数来释放
          },
          big_tensor.options());

      state_dict[big_tensor_key] = final_big_tensor;

      LOG(INFO) << "Created big_tensor for group " << group_info.group_id
                << " with shape [" << group_info.tensors.size() << ", ...]"
                << " size " << (group_info.total_size / 1024 / 1024) << " MB";
    }
  }

  LOG(INFO) << "Created " << groups_info.size() << " big_tensors from "
            << name_groups.size() << " groups (cached version)";

  return state_dict;
}

// 主动释放所有缓存的 group 内存映射
void ReleaseCachedGroupMemory() {
  std::lock_guard<std::mutex> lock(cache_mutex);
  
  LOG(INFO) << "[ReleaseCachedGroupMemory] Releasing " << group_memory_cache.size() 
            << " cached group memory mappings";
  
  for (auto& [group_key, cached] : group_memory_cache) {
    // 检查引用计数，如果还有引用，打印警告但不强制释放
    int ref_count = cached.ref_count->load();
    if (ref_count > 0) {
      LOG(INFO) << "[ReleaseCachedGroupMemory] WARNING: Group key " << group_key 
                << " still has " << ref_count << " references, skipping release";
      continue;
    }
    
    // 释放内存映射
    if (cached.contiguous_memory != nullptr) {
      munmap(cached.contiguous_memory, cached.total_size);
      LOG(INFO) << "[ReleaseCachedGroupMemory] Released memory for group key: " 
                << group_key << " (size: " << (cached.total_size / 1024 / 1024) << " MB)";
    }
    
    // 从引用计数映射中移除
    cached_memory_ref_counts.erase(cached.contiguous_memory);
  }
  
  // 清空缓存
  group_memory_cache.clear();
  
  LOG(INFO) << "[ReleaseCachedGroupMemory] All cached group memory mappings released";
}
