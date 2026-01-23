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
#include <torch/extension.h>

#include "checkpoint.h"
#include "cuda_memcpy.h"

namespace py = pybind11;

// define pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("save_tensors", &SaveTensors, "Save a state dict")
      .def("restore_tensors", &RestoreTensors, "Restore a state dict")
      .def("restore_tensors2", &RestoreTensors2, "Restore a state dict and not released")
      .def("restore_tensors_from_pinned_memory", &RestoreTensorsFromPinnedMemory, "Restore a state dict from pinned memory")
      .def("restore_tensors_from_shared_memory_names", &RestoreTensorsFromSharedMemoryNames, 
           py::arg("shm_names"), py::arg("tensor_metadata"), py::arg("chunk_size"),
           "Restore a state dict from shared memory names")
      .def("restore_experts_tensor_from_shared_memory", &RestoreExpertsFromSharedMemory,
           py::arg("shm_names"), py::arg("tensor_metadata"), py::arg("chunk_size"),
           py::arg("name_continuous_space"),
           "Restore some experts in a layer from shared memory to contiguous address space"
      )
      .def("restore_experts_groups_from_shared_memory", &RestoreExpertsGroupsFromSharedMemory,
           py::arg("shm_names"), py::arg("tensor_metadata"), py::arg("chunk_size"),
           py::arg("name_groups"),
           "Restore multiple groups of experts, each group creates a big_tensor"
      )
      .def("restore_experts_groups_from_shared_memory_profiled", &RestoreExpertsGroupsFromSharedMemoryProfiled,
           py::arg("shm_names"), py::arg("tensor_metadata"), py::arg("chunk_size"),
           py::arg("name_groups"),
           py::call_guard<py::gil_scoped_release>(),
           "Restore multiple groups of experts with performance profiling (outputs to stderr)"
      )
      .def("restore_experts_groups_from_shared_memory_silent", &RestoreExpertsGroupsFromSharedMemorySilent,
           py::arg("shm_names"), py::arg("tensor_metadata"), py::arg("chunk_size"),
           py::arg("name_groups"),
           py::call_guard<py::gil_scoped_release>(),
           "Restore multiple groups of experts silently (no output, same functionality as profiled version)"
      )
      // 缓存版本：接受 TensorIndexResizeMapCache 对象，避免重复转换
      .def("restore_experts_groups_from_shared_memory_profiled_cached", &RestoreExpertsGroupsFromSharedMemoryProfiledCached,
           py::arg("shm_names"), py::arg("tensor_metadata_cache"), py::arg("chunk_size"),
           py::arg("name_groups"),
           py::call_guard<py::gil_scoped_release>(),
           "Restore multiple groups of experts with cached tensor_metadata (avoids repeated Python->C++ conversion)"
      )
      .def("restore_experts_groups_from_shared_memory_silent_cached", &RestoreExpertsGroupsFromSharedMemorySilentCached,
           py::arg("shm_names"), py::arg("tensor_metadata_cache"), py::arg("chunk_size"),
           py::arg("name_groups"),
           py::call_guard<py::gil_scoped_release>(),
           "Restore multiple groups of experts silently with cached tensor_metadata (no output, avoids repeated Python->C++ conversion)"
      )
      // TensorIndexResizeMapCache 类，用于缓存转换后的 tensor_metadata
      .def("create_tensor_index_cache", [](const TensorIndexResizeMap& metadata) {
        return std::make_shared<TensorIndexResizeMapCache>(metadata);
      }, py::arg("tensor_metadata"),
      "Create a cached TensorIndexResizeMap to avoid repeated Python->C++ conversion")
      .def("restore_experts_groups_from_shared_memory_profiled_cached_ptr", 
           [](const std::vector<std::string>& shm_names,
              const std::shared_ptr<TensorIndexResizeMapCache>& tensor_metadata_cache,
              size_t chunk_size,
              const std::vector<std::vector<std::string>>& name_groups) {
             return RestoreExpertsGroupsFromSharedMemoryProfiledCached(
                 shm_names, *tensor_metadata_cache, chunk_size, name_groups);
           },
           py::arg("shm_names"), py::arg("tensor_metadata_cache"), py::arg("chunk_size"),
           py::arg("name_groups"),
           py::call_guard<py::gil_scoped_release>(),
           "Restore multiple groups of experts with shared_ptr cached tensor_metadata"
      )
      .def("restore_experts_groups_from_shared_memory_silent_cached_ptr", 
           [](const std::vector<std::string>& shm_names,
              const std::shared_ptr<TensorIndexResizeMapCache>& tensor_metadata_cache,
              size_t chunk_size,
              const std::vector<std::vector<std::string>>& name_groups) {
             return RestoreExpertsGroupsFromSharedMemorySilentCached(
                 shm_names, *tensor_metadata_cache, chunk_size, name_groups);
           },
           py::arg("shm_names"), py::arg("tensor_metadata_cache"), py::arg("chunk_size"),
           py::arg("name_groups"),
           py::call_guard<py::gil_scoped_release>(),
           "Restore multiple groups of experts silently with shared_ptr cached tensor_metadata (no output)"
      );
  
  // 注册 TensorIndexResizeMapCache 类
  py::class_<TensorIndexResizeMapCache, std::shared_ptr<TensorIndexResizeMapCache>>(m, "TensorIndexResizeMapCache")
      .def(py::init<const TensorIndexResizeMap&>(), py::arg("tensor_metadata"),
           "Create a cached TensorIndexResizeMap from Python dict (conversion happens only once)");
  
  // 继续注册模块级别的函数
  m.def("restore_experts_groups_from_shared_memory_cached", &RestoreExpertsGroupsFromSharedMemoryCached,
        py::arg("shm_names"), py::arg("tensor_metadata"), py::arg("chunk_size"),
        py::arg("name_groups"),
        py::call_guard<py::gil_scoped_release>(),
        "Restore multiple groups of experts with memory caching (reuses mapped memory)"
       )
      .def("release_cached_group_memory", &ReleaseCachedGroupMemory,
           "Release all cached group memory mappings (only releases if ref count is 0)"
      )
      .def("allocate_cuda_memory", &AllocateCudaMemory, "Allocate cuda memory")
      .def("free_cuda_memory", &FreeCudaMemory, "Free cuda memory")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, void*>& memory_ptrs) {
            std::unordered_map<int, std::string> memory_handles =
                GetCudaMemoryHandles(memory_ptrs);

            std::unordered_map<int, py::bytes> py_memory_handles;
            for (const auto& kv : memory_handles) {
              py_memory_handles[kv.first] = py::bytes(kv.second);
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, std::vector<void*>>& memory_ptrs) {
            auto memory_handles = GetCudaMemoryHandles(memory_ptrs);

            std::unordered_map<int, std::vector<py::bytes>> py_memory_handles;
            for (const auto& kv : memory_handles) {
              std::vector<py::bytes> handles;
              for (const auto& handle : kv.second) {
                handles.push_back(py::bytes(handle));
              }
              py_memory_handles[kv.first] = handles;
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles")
      .def(
          "get_cuda_memory_handles",
          [](const std::unordered_map<int, std::vector<uint64_t>>&
                 memory_ptrs) {
            std::unordered_map<int, std::vector<void*>> memory_ptrs_void;
            for (const auto& kv : memory_ptrs) {
              std::vector<void*> ptrs;
              for (const auto& ptr : kv.second) {
                ptrs.push_back(reinterpret_cast<void*>(ptr));
              }
              memory_ptrs_void[kv.first] = ptrs;
            }
            auto memory_handles = GetCudaMemoryHandles(memory_ptrs_void);

            std::unordered_map<int, std::vector<py::bytes>> py_memory_handles;
            for (const auto& kv : memory_handles) {
              std::vector<py::bytes> handles;
              for (const auto& handle : kv.second) {
                handles.push_back(py::bytes(handle));
              }
              py_memory_handles[kv.first] = handles;
            }
            return py_memory_handles;
          },
          py::arg("memory_ptrs"), "Get cuda memory handles")
      .def("get_device_uuid_map", &GetDeviceUuidMap, "Get device uuid map")
      
      // CUDA memory copy functions
      .def("cuda_memcpy_h2d", &CudaMemcpyTensorHostToDevice, 
           py::arg("dst_tensor"), py::arg("src_tensor"), py::arg("non_blocking") = false,
           "Copy tensor from host to device using cudaMemcpy")
      .def("cuda_memcpy_d2h", &CudaMemcpyTensorDeviceToHost,
           py::arg("dst_tensor"), py::arg("src_tensor"), py::arg("non_blocking") = false,
           "Copy tensor from device to host using cudaMemcpy")
      .def("cuda_memcpy_d2d", &CudaMemcpyTensorDeviceToDevice,
           py::arg("dst_tensor"), py::arg("src_tensor"), py::arg("non_blocking") = false,
           "Copy tensor from device to device using cudaMemcpy")
      .def("cuda_memcpy_smart", &CudaMemcpyTensorSmart,
           py::arg("dst_tensor"), py::arg("src_tensor"), py::arg("non_blocking") = false,
           "Smart tensor copy that automatically determines copy direction");
}