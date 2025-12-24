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
#include "model.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <iomanip>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

// Third-party library headers
#include <cuda_runtime.h>
// Note: glog is already included above (before model.h)
// model.h handles torch header conflicts and restores glog's LOG macro

#include "error_handling.h"

int Model::Initialize(const std::filesystem::path storage_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ != MemoryState::UNINITIALIZED) {
    return 0;
  }
  model_size_ = 0;
  partition_sizes_.clear();
  partition_paths_.clear();
  // Attempt to read from 0 until the file is not found
  for (int partition_id = 0;; ++partition_id) {
    auto tensor_path = storage_path / model_path_ /
                       ("tensor.data_" + std::to_string(partition_id));
    if (access(tensor_path.c_str(), F_OK) == -1) {
      LOG(INFO) << "Tensor file " << tensor_path << " does not exist";
      break;
    }
    struct stat st;
    if (stat(tensor_path.c_str(), &st) != 0) {
      LOG(ERROR) << "Failed to get file size of " << tensor_path;
      return -1;
    }
    model_size_ += st.st_size;
    partition_sizes_.push_back(st.st_size);
    partition_paths_.push_back(tensor_path);
  }
  if (model_size_ == 0) {
    LOG(ERROR) << "Model " << model_path_ << " does not exist";
    return -1;
  }
  state_ = MemoryState::UNALLOCATED;

  return 0;
}

int Model::ToHost(int num_threads) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ != MemoryState::ALLOCATED) {
    if (state_ == MemoryState::LOADING || state_ == MemoryState::LOADED) {
      LOG(INFO) << "Model " << model_path_ << " is already at state " << state_
                << ", skipping ToHost";
      return 0;
    } else {
      LOG(ERROR) << "Model " << model_path_ << " is at state " << state_;
      return -1;
    }
  }

  LOG(INFO) << "ToHost: Starting to load model " << model_path_
            << " size " << model_size_ << " with " << num_threads << " threads";
  std::vector<int> file_descriptors;
  // Attempt to read from 0 until the file is not found
  for (int partition_id = 0; partition_id < partition_sizes_.size();
       ++partition_id) {
    auto tensor_path = partition_paths_[partition_id];
    if (access(tensor_path.c_str(), F_OK) == -1) {
      LOG(ERROR) << "File " << tensor_path << " does not exist";
      return -1;
    }

    // Open file
    int fd = open(tensor_path.c_str(), O_DIRECT | O_RDONLY);
    if (fd < 0) {
      std::string err = "open() failed for file: " + tensor_path.string() +
                        ", error: " + strerror(errno);
      LOG(ERROR) << err;
      return -1;
    }

    file_descriptors.push_back(fd);
  }

  LOG(INFO) << "Loading model " << model_path_ << " size " << model_size_
            << " to host";
  if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "CPU memory not allocated";
    return 1;
  }

  auto host_buffers = pinned_mem_->get();
  size_t num_chunks = pinned_mem_->num_chunks();
  size_t chunk_size = pinned_mem_->chunk_size();
  host_ptr_vector_ = std::make_shared<BatchVector>();
  host_ptr_vector_->init("queue_name", num_chunks);
  std::vector<std::future<int>> futures;
  size_t chunk_per_thread = (num_chunks + num_threads - 1) / num_threads;
  LOG(INFO) << "Loading model " << model_path_ << " to host with "
            << num_threads << " threads, " << num_chunks << " chunks, "
            << chunk_size << " chunk size, " << chunk_per_thread
            << " chunks per thread";

  // Progress tracking
  std::atomic<size_t> completed_chunks{0};
  std::atomic<bool> progress_thread_running{true};
  
  // Start progress display thread
  std::thread progress_thread([&]() {
    while (progress_thread_running.load()) {
      size_t completed = completed_chunks.load();
      double progress = (double)completed / num_chunks * 100.0;
      size_t bytes_processed = completed * chunk_size;
      size_t total_bytes = num_chunks * chunk_size;
      
      // Create progress bar
      int bar_width = 50;
      int pos = (int)(progress / 100.0 * bar_width);
      std::string progress_bar = "[";
      for (int i = 0; i < bar_width; ++i) {
        if (i < pos) progress_bar += "=";
        else if (i == pos) progress_bar += ">";
        else progress_bar += " ";
      }
      progress_bar += "]";
      
      LOG(INFO) << "Loading progress: " << progress_bar 
                << " " << std::fixed << std::setprecision(1) << progress << "%"
                << " (" << completed << "/" << num_chunks << " chunks)"
                << " " << (bytes_processed / 1024 / 1024) << "MB/" 
                << (total_bytes / 1024 / 1024) << "MB";
      
      if (completed >= num_chunks) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  });

  state_ = MemoryState::LOADING;
  lock.unlock();

  for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    futures.emplace_back(std::async(std::launch::async, [&, thread_idx]() {
      size_t partition_id = 0;
      size_t file_offset = thread_idx * chunk_per_thread * chunk_size;
      while (partition_id < partition_sizes_.size() &&
             file_offset >= partition_sizes_.at(partition_id)) {
        file_offset -= partition_sizes_.at(partition_id);
        partition_id += 1;
      }
      if (partition_id >= partition_sizes_.size()) {
        LOG(INFO) << "Thread " << thread_idx << " early exits";
        return 0;
      }
      LOG(INFO) << "Thread " << thread_idx << " starting from partition "
                << partition_id << " offset " << file_offset;
      for (size_t chunk_idx = thread_idx * chunk_per_thread;
           chunk_idx < (thread_idx + 1) * chunk_per_thread &&
           chunk_idx < num_chunks;
           ++chunk_idx) {


        size_t size =
          std::min(chunk_size, model_size_ - chunk_idx * chunk_size);
        
        if (host_buffers[chunk_idx] == nullptr) {
          LOG(ERROR) << "Host buffer not allocated";
          return -1;
        }

        if (state_ == MemoryState::CANCELLED) {
          LOG(INFO) << "Loading from disk for model " << model_path_
                    << " is cancelled";
          return 0;
        }

        int fd = file_descriptors[partition_id];
        ssize_t ret =
            pread(fd, (void*)host_buffers[chunk_idx], size, file_offset);
        if (ret < 0) {
          auto tensor_path = partition_paths_[partition_id];
          LOG(ERROR) << "pread() failed for file: " << tensor_path
                     << ", error: " << strerror(errno);
          return -1;
        } else if (ret != size) {
          if (ret < size && partition_id + 1 < file_descriptors.size()) {
            partition_id += 1;
            file_offset = 0;
            size_t remaining_size = size - ret;
            int fd = file_descriptors[partition_id];
            ret = pread(fd, (void*)(host_buffers[chunk_idx] + ret),
                        remaining_size, file_offset);
            if (ret != remaining_size) {
              auto tensor_path = partition_paths_[partition_id];
              LOG(ERROR) << "Failed to read file: " << tensor_path
                         << " read: " << ret << " expected: " << remaining_size;
              return -1;
            }
          } else {
            auto tensor_path = partition_paths_[partition_id];
            LOG(ERROR) << "Failed to read file: " << tensor_path
                       << " read: " << ret << " expected: " << size;
            return -1;
          }
        }
        file_offset += ret;

        host_ptr_vector_->enqueue(chunk_idx, Batch{chunk_idx, size});
        
        // Update progress
        completed_chunks.fetch_add(1);
      }

      return 0;
    }));
  }

  bool error = false;
  for (auto& future : futures) {
    int ret = future.get();
    if (ret != 0) {
      LOG(ERROR) << "Error reading from disk, ret " << ret;
      error = true;
    }
  }
  
  // Stop progress thread
  progress_thread_running.store(false);
  if (progress_thread.joinable()) {
    progress_thread.join();
  }

  // close file
  for (int fd : file_descriptors) {
    close(fd);
  }

  lock.lock();
  if (error) {
    state_ = MemoryState::INTERRUPTED;
    // Deal with gpu replicas
    for (auto& [replica_uuid, gpu_replica] : gpu_replicas_) {
      if (gpu_replica->state_ == MemoryState::LOADING) {
        gpu_replica->state_ = MemoryState::CANCELLED;
        gpu_replica->cv_.notify_all();
      }
      // wait for gpu replicas to finish
      gpu_replica->cv_.wait(lock, [&gpu_replica] {
        return gpu_replica->state_ == MemoryState::LOADED ||
               gpu_replica->state_ == MemoryState::INTERRUPTED;
      });
      // Note: gpu replicas will be handled by the caller
    }
    // release pinned memory
    pinned_mem_.reset();
    state_ = MemoryState::UNALLOCATED;

    return -1;
  }

  state_ = MemoryState::LOADED;
  LOG(INFO) << "Finished loading model " << model_path_ << " from disk";

  return 0;
}

int Model::ToHostResize(int num_threads) {
  std::unique_lock<std::mutex> lock(mutex_);

  if (state_ != MemoryState::ALLOCATED) {
    if (state_ == MemoryState::LOADING || state_ == MemoryState::LOADED) {
      LOG(INFO) << "Model " << model_path_ << " is already at state " << state_
                << ", skipping ToHostResize";
      return 0;
    } else {
      LOG(ERROR) << "Model " << model_path_ << " is at state " << state_;
      return -1;
    }
  }

  if (tensor_index_.empty() || tensor_index_resize_.empty()) {
    LOG(ERROR) << "tensor_index_ or tensor_index_resize_ is empty";
    return -1;
  }

  LOG(INFO) << "ToHostResize: Starting to load model " << model_path_
            << " model_size_=" << model_size_ 
            << " model_size_resize_=" << model_size_resize_
            << " with " << num_threads << " threads";
  
  if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "CPU memory not allocated";
    return 1;
  }

  auto& host_buffers = pinned_mem_->get();
  size_t chunk_size = pinned_mem_->chunk_size();

  // 打开所有分区文件
  std::vector<int> file_descriptors;
  for (int partition_id = 0; partition_id < partition_sizes_.size();
       ++partition_id) {
    auto tensor_path = partition_paths_[partition_id];
    if (access(tensor_path.c_str(), F_OK) == -1) {
      LOG(ERROR) << "File " << tensor_path << " does not exist";
      return -1;
    }

    int fd = open(tensor_path.c_str(), O_DIRECT | O_RDONLY);
    if (fd < 0) {
      std::string err = "open() failed for file: " + tensor_path.string() +
                        ", error: " + strerror(errno);
      LOG(ERROR) << err;
      return -1;
    }
    file_descriptors.push_back(fd);
  }

  LOG(INFO) << "Loading " << tensor_index_resize_.size()
            << " tensors from disk to host with resized layout";

  // 按 chunk 分组 tensors，以便多线程处理
  size_t num_chunks = pinned_mem_->num_chunks();
  std::vector<std::vector<std::string>> chunk_tensors(num_chunks);
  
  for (const auto& [name, resize_info] : tensor_index_resize_) {
    if (tensor_index_.find(name) == tensor_index_.end()) {
      continue;
    }
    auto [resize_offset, resize_size, resize_shape, resize_strides,
          resize_dtype] = resize_info;
    
    // 计算 tensor 跨越的 chunks
    std::vector<std::tuple<int, size_t, size_t>> chunks =
        MapDataToChunks(resize_offset, resize_size, chunk_size);
    
    for (const auto& [chunk_id, chunk_offset, chunk_size_in_chunk] : chunks) {
      if (chunk_id >= 0 && static_cast<size_t>(chunk_id) < num_chunks) {
        chunk_tensors[chunk_id].push_back(name);
      }
    }
  }

  host_ptr_vector_ = std::make_shared<BatchVector>();
  host_ptr_vector_->init("queue_name", num_chunks);
  std::vector<std::future<int>> futures;
  size_t chunk_per_thread = (num_chunks + num_threads - 1) / num_threads;
  LOG(INFO) << "Loading model " << model_path_ << " to host with "
            << num_threads << " threads, " << num_chunks << " chunks, "
            << chunk_size << " chunk size, " << chunk_per_thread
            << " chunks per thread";

  // Progress tracking
  std::atomic<size_t> completed_chunks{0};
  std::atomic<bool> progress_thread_running{true};
  
  // Start progress display thread
  std::thread progress_thread([&]() {
    LOG(INFO) << "ToHostResize: Progress thread started, monitoring " << num_chunks << " chunks";
    size_t last_completed = 0;
    int iteration = 0;
    while (progress_thread_running.load()) {
      size_t completed = completed_chunks.load();
      double progress = num_chunks > 0 ? (double)completed / num_chunks * 100.0 : 0.0;
      size_t bytes_processed = completed * chunk_size;
      size_t total_bytes = num_chunks * chunk_size;
      
      // 每次迭代都输出，或者当进度有变化时输出
      if (completed != last_completed || iteration == 0) {
        // Create progress bar
        int bar_width = 50;
        int pos = num_chunks > 0 ? (int)(progress / 100.0 * bar_width) : 0;
        std::string progress_bar = "[";
        for (int i = 0; i < bar_width; ++i) {
          if (i < pos) progress_bar += "=";
          else if (i == pos) progress_bar += ">";
          else progress_bar += " ";
        }
        progress_bar += "]";
        
        LOG(INFO) << "ToHostResize loading progress: " << progress_bar 
                  << " " << std::fixed << std::setprecision(1) << progress << "%"
                  << " (" << completed << "/" << num_chunks << " chunks)"
                  << " " << (bytes_processed / 1024 / 1024) << "MB/" 
                  << (total_bytes / 1024 / 1024) << "MB";
        last_completed = completed;
      }
      
      if (completed >= num_chunks) {
        LOG(INFO) << "ToHostResize: Progress thread finished, all " << num_chunks << " chunks completed";
        break;
      }
      iteration++;
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  });

  state_ = MemoryState::LOADING;
  lock.unlock();

  for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    futures.emplace_back(std::async(std::launch::async, [&, thread_idx]() {
      for (size_t chunk_idx = thread_idx * chunk_per_thread;
           chunk_idx < (thread_idx + 1) * chunk_per_thread &&
           chunk_idx < num_chunks;
           ++chunk_idx) {
        
        if (state_ == MemoryState::CANCELLED) {
          LOG(INFO) << "Loading from disk for model " << model_path_
                    << " is cancelled";
          return 0;
        }

        // 处理该 chunk 中的所有 tensors
        std::set<std::string> processed_tensors;  // 避免重复处理
        for (const auto& name : chunk_tensors[chunk_idx]) {
          if (processed_tensors.find(name) != processed_tensors.end()) {
            continue;
          }
          processed_tensors.insert(name);

          // 从 tensor_index_ 获取原始文件中的位置
          if (tensor_index_.find(name) == tensor_index_.end()) {
            continue;
          }

          auto [resize_offset, resize_size, resize_shape, resize_strides,
                resize_dtype] = tensor_index_resize_[name];
          auto [file_offset, file_size, file_shape, file_strides, file_dtype] =
              tensor_index_[name];

          if (file_size != resize_size) {
            LOG(ERROR) << "Tensor " << name << " size mismatch: file_size="
                       << file_size << ", resize_size=" << resize_size;
            continue;
          }

          // 计算 tensor 在 resize 布局中跨越的 chunks
          std::vector<std::tuple<int, size_t, size_t>> resize_chunks =
              MapDataToChunks(resize_offset, resize_size, chunk_size);

          // 处理每个 chunk 部分
          size_t remaining_size = resize_size;
          size_t file_read_offset = 0;

          for (const auto& [resize_chunk_id, resize_chunk_offset,
                            resize_chunk_size] : resize_chunks) {
            if (resize_chunk_id < 0 ||
                static_cast<size_t>(resize_chunk_id) >= num_chunks) {
              LOG(ERROR) << "Tensor " << name << " invalid chunk_id "
                         << resize_chunk_id;
              return -1;
            }

            // 找到包含该 offset 的分区
            size_t partition_id = 0;
            size_t partition_file_offset = file_offset + file_read_offset;
            while (partition_id < partition_sizes_.size() &&
                   partition_file_offset >= partition_sizes_[partition_id]) {
              partition_file_offset -= partition_sizes_[partition_id];
              partition_id++;
            }

            if (partition_id >= partition_sizes_.size()) {
              LOG(ERROR) << "Tensor " << name << " offset " << file_offset
                         << " exceeds file size";
              return -1;
            }

            // 从文件读取到 pinned memory
            int fd = file_descriptors[partition_id];
            void* dst_ptr = host_buffers[resize_chunk_id] + resize_chunk_offset;
            ssize_t ret = pread(fd, (void*)dst_ptr, resize_chunk_size,
                                partition_file_offset);
            if (ret != static_cast<ssize_t>(resize_chunk_size)) {
              LOG(ERROR) << "Failed to read tensor " << name
                         << " from file: read " << ret << " expected "
                         << resize_chunk_size;
              return -1;
            }

            file_read_offset += resize_chunk_size;
            remaining_size -= resize_chunk_size;
          }
        }

        host_ptr_vector_->enqueue(chunk_idx, Batch{chunk_idx, chunk_size});
        
        // Update progress
        size_t current = completed_chunks.fetch_add(1) + 1;
        if (current % 10 == 0 || current == num_chunks) {
          LOG(INFO) << "ToHostResize: Completed " << current << "/" << num_chunks << " chunks";
        }
      }

      return 0;
    }));
  }

  bool error = false;
  for (auto& future : futures) {
    int ret = future.get();
    if (ret != 0) {
      LOG(ERROR) << "Error reading from disk, ret " << ret;
      error = true;
    }
  }
  
  // Stop progress thread
  progress_thread_running.store(false);
  if (progress_thread.joinable()) {
    progress_thread.join();
  }

  // close file
  for (int fd : file_descriptors) {
    close(fd);
  }

  lock.lock();
  if (error) {
    state_ = MemoryState::INTERRUPTED;
    // Deal with gpu replicas
    for (auto& [replica_uuid, gpu_replica] : gpu_replicas_) {
      if (gpu_replica->state_ == MemoryState::LOADING) {
        gpu_replica->state_ = MemoryState::CANCELLED;
        gpu_replica->cv_.notify_all();
      }
      // wait for gpu replicas to finish
      gpu_replica->cv_.wait(lock, [&gpu_replica] {
        return gpu_replica->state_ == MemoryState::LOADED ||
               gpu_replica->state_ == MemoryState::INTERRUPTED;
      });
      // Note: gpu replicas will be handled by the caller
    }
    // release pinned memory
    pinned_mem_.reset();
    state_ = MemoryState::UNALLOCATED;

    return -1;
  }

  state_ = MemoryState::LOADED;
  LOG(INFO) << "Finished loading model " << model_path_ << " from disk with resized layout";

  return 0;
}

int Model::ToGpu(
    const std::string& replica_uuid, const MemPtrListMap& device_ptrs,
    const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
    const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles,
    const bool& use_fixed_gpu_ptrs) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(ERROR) << "Model " << model_path_ << " is not initialized";
    return -1;
  }

  if (gpu_replicas_.find(replica_uuid) != gpu_replicas_.end()) {
    LOG(ERROR) << "Replica " << replica_uuid << " already exists";
    return -1;
  }
  LOG(INFO) << "Creating replica " << replica_uuid;
  gpu_replicas_.emplace(replica_uuid, std::make_shared<GpuReplica>());
  GpuReplicaPtr gpu_replica = gpu_replicas_.at(replica_uuid);
  for (const auto& [device_id, _] : device_ptrs) {
    LOG(INFO) << "Creating queue for device " << device_id;
    gpu_replica->gpu_loading_queue_.emplace(device_id,
                                            std::make_shared<BatchQueue>());
  }
  gpu_replica->device_ptrs_ = device_ptrs;
  gpu_replica->state_ = MemoryState::LOADING;
  LOG(INFO) << "Created replica " << replica_uuid;
  cv_.notify_all();
  lock.unlock();

  // Start a dispatcher first
  auto dispatch_future = std::async(
      std::launch::async,
      [this, gpu_replica, mem_copy_chunks, mem_copy_handles]() {
        return DispatchToGpu(gpu_replica, mem_copy_chunks, mem_copy_handles);
      });

  LOG(INFO) << "Dispatcher started for model " << model_path_;

  std::unordered_map<int, std::future<int>> futures;
  for (auto& [device_id, device_ptr_list] : device_ptrs) {
    futures.emplace(
        device_id, std::async(std::launch::async, [this, gpu_replica, device_id,
                                                   device_ptr_list]() {
          auto gpu_loading_queue =
              gpu_replica->gpu_loading_queue_.at(device_id);
          if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
            LOG(ERROR) << "CPU memory not allocated";
            return 1;
          }

          cudaError_t err = cudaSetDevice(device_id);
          if (err != cudaSuccess) {
            LOG(ERROR) << "Error setting device " << cudaGetErrorString(err);
            return 1;
          }

          auto& host_buffers = pinned_mem_->get();

          size_t loaded_size = 0;
          while (true) {
            auto [chunk_id, chunk_offset, size, gpu_offset, handle_idx] =
                gpu_loading_queue->dequeue();
            if (size == 0) {
              break;
            }
            if (gpu_replica->state_ == MemoryState::CANCELLED) {
              LOG(INFO) << "Loading from mem for model " << model_path_
                        << " is cancelled,"
                        << " chunk " << chunk_id << " offset "
                        << " size " << size;
              return 0;
            }

            CUDA_CHECK(
                cudaMemcpy(
                    (void*)((char*)device_ptr_list[handle_idx] + gpu_offset),
                    (void*)(host_buffers[chunk_id] + chunk_offset), size,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy Error");
            loaded_size += size;
          }

          LOG(INFO) << "Finished loading tensor from memory to device "
                    << device_id;

          return 0;
        }));
  }

  LOG(INFO) << "Waiting for model " << model_path_ << " num tasks "
            << futures.size() << " state " << gpu_replica->state_;
  dispatch_future.wait();
  bool error = false;
  for (auto& [device_id, future] : futures) {
    int ret = future.get();
    if (ret != 0) {
      LOG(ERROR) << "Error copying to device " << device_id;
      error = true;
    }
  }

  lock.lock();
  futures.clear();

  if (error) {
    LOG(ERROR) << "Failed to load model " << model_path_;
    gpu_replica->state_ = MemoryState::INTERRUPTED;
  } else {
    gpu_replica->state_ = MemoryState::LOADED;
  }
  gpu_replica->cv_.notify_all();

  // release memory handles, if use_fixed_gpu_ptrs is false, we need to release the memory handles, else we not need
  if (!use_fixed_gpu_ptrs){
    // TODO: move to background thread
    for (auto& [device_id, device_ptr_list] : gpu_replica->device_ptrs_) {
      cudaSetDevice(device_id);
      for (auto device_ptr : device_ptr_list) {
        cudaError_t err = cudaIpcCloseMemHandle(device_ptr);
        if (err != cudaSuccess) {
          LOG(ERROR) << "Failed to close memory handle for device " << device_id
                    << " error: " << cudaGetErrorString(err);
        }
      }
    }
  }

  if (gpu_replica->state_ == MemoryState::INTERRUPTED) {
    LOG(ERROR) << "Model " << model_path_ << " replica " << replica_uuid
               << " is interrupted";
    return -1;
  }

  return 0;
}

int Model::WaitInHost() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ < MemoryState::LOADED) {
    cv_.wait(lock, [this] {
      return state_ == MemoryState::LOADED ||
             state_ == MemoryState::INTERRUPTED;
    });
  }

  if (state_ >= MemoryState::INTERRUPTED) {
    LOG(INFO) << "Model " << model_path_ << " is interrupted";
    return 1;
  }

  return 0;
}

int Model::WaitInGpu(const std::string& replica_uuid) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (gpu_replicas_.find(replica_uuid) == gpu_replicas_.end()) {
    cv_.wait(lock, [this, replica_uuid] {
      return gpu_replicas_.find(replica_uuid) != gpu_replicas_.end();
    });
  }

  auto& gpu_replica = gpu_replicas_.at(replica_uuid);

  if (gpu_replica->state_ < MemoryState::LOADED) {
    gpu_replica->cv_.wait(lock, [&gpu_replica] {
      return gpu_replica->state_ == MemoryState::LOADED ||
             gpu_replica->state_ == MemoryState::INTERRUPTED;
    });
  }

  if (gpu_replica->state_ >= MemoryState::INTERRUPTED) {
    LOG(INFO) << "Model " << model_path_ << " is interrupted";
    return 1;
  }

  return 0;
}

int Model::FreeGpu(const std::string& replica_uuid) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (gpu_replicas_.find(replica_uuid) == gpu_replicas_.end()) {
    LOG(ERROR) << "Model " << model_path_ << " replica " << replica_uuid
               << " is not registered";
    return -1;
  }

  auto& gpu_replica = gpu_replicas_.at(replica_uuid);
  if (gpu_replica->state_ == MemoryState::UNINITIALIZED) {
    LOG(INFO) << "Model " << model_path_ << " replica " << replica_uuid
                 << " is not initialized";
    gpu_replicas_.erase(replica_uuid);
    return 0;
  }

  if (gpu_replica->state_ == MemoryState::LOADING) {
    LOG(INFO) << "Waiting for model " << model_path_ << " replica "
              << replica_uuid << " to be loaded";
    gpu_replica->cv_.wait(lock, [&gpu_replica] {
      return gpu_replica->state_ == MemoryState::LOADED ||
             gpu_replica->state_ == MemoryState::INTERRUPTED;
    });
  }

  gpu_replicas_.erase(replica_uuid);
  return 0;
}

int Model::FreeHost() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(INFO) << "Model " << model_path_ << " is not initialized";
    return 1;
  }

  if (state_ == MemoryState::UNALLOCATED) {
    LOG(INFO) << "Model " << model_path_ << " is not allocated";
    return 1;
  }

  if (state_ == MemoryState::LOADING) {
    LOG(INFO) << "Waiting for model " << model_path_ << " to be loaded";
    cv_.wait(lock, [this] {
      return state_ == MemoryState::LOADED ||
             state_ == MemoryState::INTERRUPTED;
    });
  }

  // make sure no gpu replicas are loading
  for (auto& [replica_uuid, gpu_replica] : gpu_replicas_) {
    if (gpu_replica->state_ == MemoryState::LOADING) {
      LOG(INFO) << "Waiting for replica " << replica_uuid << " to be loaded";
      gpu_replica->cv_.wait(lock, [&gpu_replica] {
        return gpu_replica->state_ == MemoryState::LOADED ||
               gpu_replica->state_ == MemoryState::INTERRUPTED;
      });
    }
  }

  // free pinned memory
  int freed_chunks = pinned_mem_->num_chunks();
  pinned_mem_.reset();
  state_ = MemoryState::UNALLOCATED;

  return 0;
}

int Model::TryFreeHost() {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(INFO) << "Model " << model_path_ << " is not initialized";
    return 0;
  }

  if (state_ == MemoryState::UNALLOCATED) {
    LOG(INFO) << "Model " << model_path_ << " is not allocated";
    return 0;
  }

  if (state_ == MemoryState::LOADING) {
    return -1;
  }

  // make sure no gpu replicas are loading
  for (auto& [replica_uuid, gpu_replica] : gpu_replicas_) {
    if (gpu_replica->state_ == MemoryState::LOADING) {
      return -1;
    }
  }

  // free pinned memory
  int freed_chunks = pinned_mem_->num_chunks();
  pinned_mem_.reset();
  state_ = MemoryState::UNALLOCATED;

  return freed_chunks;
}

int Model::DispatchToGpu(
    const std::shared_ptr<GpuReplica>& gpu_replica,
    const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
    const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles) {
  // device_id, chunk_offset, size, gpu_offset

  size_t num_chunks = pinned_mem_->num_chunks();
  std::vector<std::vector<GpuChunk>> chunk_id_to_gpu_chunks(num_chunks);
  for (const auto& [device_id, mem_copy_chunk_list] : mem_copy_chunks) {
    const auto& device_handles = mem_copy_handles.at(device_id);
    std::vector<size_t> handle_offsets(device_handles.size(), 0);

    for (auto [host_offset, size, gpu_offset, handle_idx] :
         mem_copy_chunk_list) {
      handle_offsets[handle_idx] = gpu_offset;

      std::vector<std::tuple<int, size_t, size_t>> chunks =
          MapDataToChunks(host_offset, size, pinned_mem_->chunk_size());
      for (const auto& [chunk_id, chunk_offset, size] : chunks) {
        chunk_id_to_gpu_chunks[chunk_id].push_back(
            std::make_tuple(device_id, chunk_offset, size,
                            handle_offsets[handle_idx], handle_idx));
        handle_offsets[handle_idx] += size;
      }
    }
  }

  for (int i = 0; i < host_ptr_vector_->capacity(); i++) {
    auto data_chunk = host_ptr_vector_->dequeue(i);
    auto chunk_id = data_chunk.chunk_id_;
    auto& gpu_chunks = chunk_id_to_gpu_chunks[chunk_id];
    for (const auto& [device_id, chunk_offset, size, gpu_offset, handle_idx] :
         gpu_chunks) {
      auto& gpu_loading_queue = gpu_replica->gpu_loading_queue_.at(device_id);
      // LOG(INFO) << "Enqueueing chunk " << chunk_id << " offset " <<
      // chunk_offset
      //           << " size " << size << " to device " << device_id;
      gpu_loading_queue->enqueue(
          GpuBatch{chunk_id, chunk_offset, size, gpu_offset, handle_idx});
    }
  }

  // notify end of loading
  for (auto& [device_id, gpu_loading_queue] : gpu_replica->gpu_loading_queue_) {
    gpu_loading_queue->enqueue(GpuBatch{});
  }

  return 0;
}

std::vector<std::tuple<int, size_t, size_t>> Model::MapDataToChunks(
    size_t offset, size_t size, size_t chunk_size) {
  int start_chunk = offset / chunk_size;
  size_t offset_in_start_chunk = offset % chunk_size;
  size_t remaining_data = size;
  std::vector<std::tuple<int, size_t, size_t>> output;

  for (int chunk_id = start_chunk; remaining_data > 0; ++chunk_id) {
    const size_t chunk_data_size =
        (chunk_id == start_chunk)
            ? std::min(chunk_size - offset_in_start_chunk, remaining_data)
            : std::min(chunk_size, remaining_data);
    output.emplace_back(chunk_id,
                        chunk_id == start_chunk ? offset_in_start_chunk : 0,
                        chunk_data_size);
    remaining_data -= chunk_data_size;
  }

  return output;
}

int Model::AllocatePinnedMemory(std::shared_ptr<PinnedMemoryPool> pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(ERROR) << "Model " << model_path_ << " is not initialized";
    return -1;
  }
  if (state_ != MemoryState::UNALLOCATED) {
    return 0;
  }
  
  // 如果使用 resize，则分配 model_size_resize_ 的大小
  size_t size_to_allocate = model_size_;
  if (!tensor_index_resize_.empty() && model_size_resize_ > 0) {
    size_to_allocate = model_size_resize_;
    LOG(INFO) << "Using resize layout, allocating " << size_to_allocate
              << " bytes (" << size_to_allocate / MB << " MB)";
  } else {
    LOG(INFO) << "Using normal layout, allocating " << size_to_allocate
              << " bytes (" << size_to_allocate / MB << " MB)";
  }
  
  pinned_mem_ = std::make_shared<PinnedMemory>();
  int ret = pinned_mem_->Allocate(size_to_allocate, pool);
  if (ret < 0) {
    LOG(ERROR) << "Error allocating CPU memory for model " << model_path_;
    return ret;
  } else if (ret > 0) {
    LOG(INFO) << "Not enough memory for model " << model_path_;
    return ret;
  } else if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "CPU memory not allocated";
    return -1;
  }
  
  state_ = MemoryState::ALLOCATED;
  return 0;
}

int Model::AllocatePinnedMemory(std::shared_ptr<PinnedMemoryPoolShared> pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(ERROR) << "Model " << model_path_ << " is not initialized";
    return -1;
  }
  if (state_ != MemoryState::UNALLOCATED) {
    return 0;
  }
  
  // 如果使用 resize，则分配 model_size_resize_ 的大小
  size_t size_to_allocate = model_size_;
  if (!tensor_index_resize_.empty() && model_size_resize_ > 0) {
    size_to_allocate = model_size_resize_;
    LOG(INFO) << "Using resize layout, allocating " << size_to_allocate
              << " bytes (" << size_to_allocate / MB << " MB)";
  } else {
    LOG(INFO) << "Using normal layout, allocating " << size_to_allocate
              << " bytes (" << size_to_allocate / MB << " MB)";
  }
  
  pinned_mem_ = std::make_shared<PinnedMemory>();
  int ret = pinned_mem_->Allocate(size_to_allocate, pool);
  if (ret < 0) {
    LOG(ERROR) << "Error allocating shared CPU memory for model " << model_path_;
    return ret;
  } else if (ret > 0) {
    LOG(INFO) << "Not enough shared memory for model " << model_path_;
    return ret;
  } else if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "Shared CPU memory not allocated";
    return -1;
  }
  
  state_ = MemoryState::ALLOCATED;
  return 0;
}

std::vector<std::string> Model::GetSharedMemoryNames() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!pinned_mem_) {
    return {};
  }
  return pinned_mem_->get_shm_names();
}

void Model::SetTensorInfo(const TensorIndexMap& tensor_index,
                          const TensorIndexResizeMap& tensor_index_resize) {
  std::lock_guard<std::mutex> lock(mutex_);
  
  // 深拷贝 tensor_index
  tensor_index_.clear();
  for (const auto& [name, info] : tensor_index) {
    auto [offset, size, shape, strides, dtype] = info;
    tensor_index_[name] = std::make_tuple(
        offset,
        size,
        std::vector<size_t>(shape),  // 深拷贝 shape vector
        std::vector<size_t>(strides),  // 深拷贝 strides vector
        std::string(dtype)  // 深拷贝 dtype string
    );
  }
  
  // 深拷贝 tensor_index_resize 并计算 model_size_resize_
  tensor_index_resize_.clear();
  uint64_t max_offset_plus_size = 0;
  for (const auto& [name, info] : tensor_index_resize) {
    auto [offset, size, shape, strides, dtype] = info;
    tensor_index_resize_[name] = std::make_tuple(
        offset,
        size,
        std::vector<size_t>(shape),  // 深拷贝 shape vector
        std::vector<size_t>(strides),  // 深拷贝 strides vector
        std::string(dtype)  // 深拷贝 dtype string
    );
    
    // 计算最终大小：找到最大的 offset + size
    uint64_t end_offset = offset + size;
    if (end_offset > max_offset_plus_size) {
      max_offset_plus_size = end_offset;
    }
  }
  model_size_resize_ = max_offset_plus_size;
  
  LOG(INFO) << "SetTensorInfo: model_size_resize_ = " << model_size_resize_
            << " bytes (" << model_size_resize_ / MB << " MB)";
}