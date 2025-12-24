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
// Include glog BEFORE torch headers to ensure glog's LOG macro takes precedence
#include <glog/logging.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>

#include "checkpoint_store.h"
#include "types_and_defs.h"
#include "sllm_store/model.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Initialize glog if not already initialized
  // This ensures glog is ready before any C++ code uses LOG macros
  static bool glog_initialized = false;
  if (!glog_initialized) {
    // Check if glog is already initialized by trying to set a flag
    // If InitGoogleLogging hasn't been called, this will initialize it
    // If it has been called, this is safe (glog handles re-initialization)
    google::InitGoogleLogging("sllm_store");
    glog_initialized = true;
    
    // Set glog to also log to stderr for immediate visibility
    FLAGS_alsologtostderr = true;
    LOG(INFO) << "glog initialized";
  }
  // Binding for MemCopyChunk
  py::class_<MemCopyChunk>(m, "MemCopyChunk")
      .def(py::init<>())
      .def_readwrite("src_offset", &MemCopyChunk::src_offset_)
      .def_readwrite("size", &MemCopyChunk::size_)
      .def_readwrite("dst_offset", &MemCopyChunk::dst_offset_)
      .def_readwrite("handle_idx", &MemCopyChunk::handle_idx_);

  py::class_<CheckpointStore>(m, "CheckpointStore")
      .def(py::init<const std::string&, size_t, int, size_t, bool, const std::string&>(),
           py::arg("storage_path"), py::arg("memory_pool_size"),
           py::arg("num_thread"), py::arg("chunk_size"),
           py::arg("use_shared_memory") = false,
           py::arg("shm_name_prefix") = "/sllm_pinned_pool")
      .def("register_model_info", &CheckpointStore::RegisterModelInfo,
           py::arg("model_path"), py::arg("tensor_index"), 
           py::arg("tensor_index_resize"),
           "Register the model information and return its size.")
      .def("load_model_from_disk_async",
           &CheckpointStore::LoadModelFromDiskAsync, py::arg("model_path"),
           "Load a model from disk asynchronously.")
      .def(
          "load_model_from_mem_async",
          [](CheckpointStore& cs, const std::string& model_path,
             const std::string& replica_uuid,
             const std::unordered_map<std::string, std::vector<py::bytes>>&
                 gpu_memory_handles,
             const MemCopyChunkListMap& mem_copy_chunks,
             const bool& use_fixed_gpu_ptrs) {
            // Convert memory handles to MemCopyHandleListMap
            MemCopyHandleListMap gpu_memory_handles_map;
            for (const auto& [device_id, handles] : gpu_memory_handles) {
              MemCopyHandleList handle_list;
              for (const auto& handle : handles) {
                handle_list.push_back(
                    MemCopyHandle{handle.cast<std::string>()});
              }
              gpu_memory_handles_map[device_id] = handle_list;
            }
            return cs.LoadModelFromMemAsync(model_path, replica_uuid,
                                            gpu_memory_handles_map,
                                            mem_copy_chunks,
                                            use_fixed_gpu_ptrs);
          },
          py::arg("model_path"), py::arg("replica_uuid"),
          py::arg("gpu_memory_handles"), py::arg("mem_copy_chunks"),
          py::arg("use_fixed_gpu_ptrs"),
          "Load a model from memory asynchronously.")
      .def("register_fixed_gpu_ptrs", 
          [](CheckpointStore& cs, const std::unordered_map<std::string, std::vector<py::bytes>>&
                 gpu_memory_handles) {
            MemCopyHandleListMap gpu_memory_handles_map;
            for (const auto& [device_id, handles] : gpu_memory_handles) {
              MemCopyHandleList handle_list;
              for (const auto& handle : handles) {
                handle_list.push_back(
                    MemCopyHandle{handle.cast<std::string>()});
              }
              gpu_memory_handles_map[device_id] = handle_list;
            }
            return cs.RegisterFixedGpuPtrs(gpu_memory_handles_map);
          },
          py::arg("gpu_memory_handles"),
          "Register fixed gpu pointers.")
      .def("release_registered_fixed_gpu_ptrs",
          [](CheckpointStore& cs, const std::unordered_map<std::string, std::vector<py::bytes>>&
                 cuda_memory_handles_need_released) {
            MemCopyHandleListMap cuda_memory_handles_need_released_map;
            for (const auto& [device_id, handles] : cuda_memory_handles_need_released) {
              MemCopyHandleList handle_list;
              for (const auto& handle : handles) {
                handle_list.push_back(
                    MemCopyHandle{handle.cast<std::string>()});
              }
              cuda_memory_handles_need_released_map[device_id] = handle_list;
            }
            return cs.ReleaseRegisteredFixedGpuPtrs(cuda_memory_handles_need_released_map);
          },
          py::arg("cuda_memory_handles_need_released"),
           "Release registered fixed gpu pointers.")
      .def("release_all_registered_fixed_gpu_ptrs",
           &CheckpointStore::ReleaseRegisteredFixedGpuPtrsAll,
           "Release all registered fixed gpu pointers.")
      .def("wait_model_in_gpu", &CheckpointStore::WaitModelInGpu,
           py::arg("model_path"), py::arg("replica_uuid"),
           "Wait for a model to be available in GPU memory.")
      .def("unload_model_from_host", &CheckpointStore::UnloadModelFromHost,
           py::arg("model_path"), "Unload a model from the host memory.")
      .def("clear_mem", &CheckpointStore::ClearMem,
           "Clear all allocated memory.")
      .def("get_mem_pool_size", &CheckpointStore::GetMemPoolSize,
           "Get the memory pool size.")
      .def("get_chunk_size", &CheckpointStore::GetChunkSize,
           "Get the chunk size.")
      .def("get_model_shared_memory_names", &CheckpointStore::GetModelSharedMemoryNames,
           py::arg("model_path"),
           "Get shared memory names for a model.")
      .def("is_using_shared_memory", &CheckpointStore::IsUsingSharedMemory,
           "Check if using shared memory pool.")
      .def("__repr__",
           [](const CheckpointStore& cs) { return "<CheckpointStore>"; });
}