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
#include <sys/stat.h>
#include <torch/extension.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "progress_bar.h"
#include "tensor_writer.h"

// for shared memory
#include <sys/mman.h>
#include <sys/shm.h>

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
        std::cerr << "Cannot find device " << device << std::endl;
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
    cudaSetDevice(device);
    cudaMalloc(&ptr, size);
    memory_ptrs[device] = ptr;
  }
  return memory_ptrs;
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
      std::cerr << "Duplicate device id: " << p.second << std::endl;
      exit(1);
    }
    device_uuid_map[p.second] = p.first;
  }
  return device_uuid_map;
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