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
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

// Third-party library headers
#include <cuda_runtime.h>
#include <glog/logging.h>

#include "error_handling.h"

void GpuReplica::PriorityGpuQueue::Push(uint64_t task_id, CopyPriority priority) {
  std::lock_guard<std::mutex> lock(mu_);
  if (closed_) {
    return;
  }
  if (priority == CopyPriority::HIGH) {
    high_queue_.push(task_id);
  } else {
    low_queue_.push(task_id);
  }
  cv_.notify_one();
}

bool GpuReplica::PriorityGpuQueue::Pop(uint64_t* task_id) {
  std::unique_lock<std::mutex> lock(mu_);
  cv_.wait(lock, [this] { return closed_ || !high_queue_.empty() || !low_queue_.empty(); });
  if (!high_queue_.empty()) {
    *task_id = high_queue_.front();
    high_queue_.pop();
    return true;
  }
  if (!low_queue_.empty()) {
    *task_id = low_queue_.front();
    low_queue_.pop();
    return true;
  }
  return false;
}

void GpuReplica::PriorityGpuQueue::Close() {
  std::lock_guard<std::mutex> lock(mu_);
  closed_ = true;
  cv_.notify_all();
}

size_t GpuReplica::PriorityGpuQueue::HighSize() {
  std::lock_guard<std::mutex> lock(mu_);
  return high_queue_.size();
}

size_t GpuReplica::PriorityGpuQueue::LowSize() {
  std::lock_guard<std::mutex> lock(mu_);
  return low_queue_.size();
}

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
      return 0;
    } else {
      LOG(ERROR) << "Model " << model_path_ << " is at state " << state_;
      return -1;
    }
  }

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
      bool retried_without_odirect = false;
      if (errno == EINVAL) {
        LOG(ERROR) << "O_DIRECT not supported on " << tensor_path
                   << ", falling back to compatible mode (may severely impact "
                      "the performance!)";
        fd = open(tensor_path.c_str(), O_RDONLY);
        retried_without_odirect = true;
      }
      if (fd < 0) {
        std::string err_msg_prefix =
            retried_without_odirect ? "open() failed for file (no O_DIRECT): "
                                    : "open() failed for file: ";
        std::string err = err_msg_prefix + tensor_path.string() +
                          ", error: " + strerror(errno);
        LOG(ERROR) << err;
        return -1;
      }
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
  size_t num_chunks = pinned_mem_->num_chunks();

  // Open all partition files.
  // Do NOT use O_DIRECT: tensor resize_offsets are not guaranteed to be
  // block-aligned, so O_DIRECT pread() would fail with EINVAL.
  std::vector<int> file_descriptors;
  for (size_t partition_id = 0; partition_id < partition_sizes_.size();
       ++partition_id) {
    auto tensor_path = partition_paths_[partition_id];
    if (access(tensor_path.c_str(), F_OK) == -1) {
      LOG(ERROR) << "File " << tensor_path << " does not exist";
      for (int fd : file_descriptors) close(fd);
      return -1;
    }
    int fd = open(tensor_path.c_str(), O_RDONLY);
    // int fd = open(tensor_path.c_str(), O_DIRECT | O_RDONLY);
    if (fd < 0) {
      LOG(ERROR) << "open() failed for file: " << tensor_path.string()
                 << ", error: " << strerror(errno);
      for (int ofd : file_descriptors) close(ofd);
      return -1;
    }
    file_descriptors.push_back(fd);
  }

  // Build an immutable flat list of tensor copy tasks in the main thread.
  // Each task records everything a worker needs; no shared mutable map access
  // in worker threads, which eliminates the data-race that caused segfaults.
  struct TensorTask {
    size_t resize_offset;  // destination byte offset in pinned buffer
    size_t file_offset;    // source byte offset across all partitions
    size_t size;           // bytes to copy (file_size == resize_size)
  };
  std::vector<TensorTask> tasks;
  tasks.reserve(tensor_index_resize_.size());

  for (const auto& [name, resize_info] : tensor_index_resize_) {
    auto it = tensor_index_.find(name);
    if (it == tensor_index_.end()) {
      LOG(WARNING) << "Tensor " << name
                   << " in tensor_index_resize_ but not in tensor_index_, skipping";
      continue;
    }
    const auto& [resize_offset, resize_size, resize_shape, resize_strides,
                 resize_dtype] = resize_info;
    const auto& [file_offset, file_size, file_shape, file_strides, file_dtype] =
        it->second;

    if (file_size != resize_size) {
      LOG(ERROR) << "Tensor " << name << " size mismatch: file_size=" << file_size
                 << " resize_size=" << resize_size << ", skipping";
      continue;
    }
    if (resize_offset + resize_size > model_size_resize_) {
      LOG(ERROR) << "Tensor " << name << " resize_offset=" << resize_offset
                 << " size=" << resize_size
                 << " exceeds model_size_resize_=" << model_size_resize_;
      continue;
    }
    tasks.push_back({resize_offset, file_offset, resize_size});
  }

  LOG(INFO) << "ToHostResize: " << tasks.size() << " tensors, "
            << num_chunks << " chunks, chunk_size=" << chunk_size;

  host_ptr_vector_ = std::make_shared<BatchVector>();
  host_ptr_vector_->init("queue_name", num_chunks);

  // Divide tensors among threads (not chunks).
  // Each tensor is owned by exactly one thread → no write-side data races on
  // host_buffers (tensors have non-overlapping resize destinations).
  size_t total_tasks = tasks.size();
  size_t tasks_per_thread = (total_tasks + num_threads - 1) / num_threads;

  std::atomic<size_t> completed_tasks{0};
  std::atomic<bool> progress_thread_running{true};

  std::thread progress_thread([&]() {
    while (progress_thread_running.load()) {
      size_t done = completed_tasks.load();
      double pct = total_tasks > 0 ? 100.0 * done / total_tasks : 0.0;
      LOG(INFO) << "ToHostResize progress: " << done << "/" << total_tasks
                << " tensors (" << std::fixed << std::setprecision(1) << pct << "%)";
      if (done >= total_tasks) break;
      std::this_thread::sleep_for(std::chrono::seconds(2));
    }
  });

  state_ = MemoryState::LOADING;
  lock.unlock();

  std::vector<std::future<int>> futures;
  for (int thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
    futures.emplace_back(
        std::async(std::launch::async, [&, thread_idx]() -> int {
          size_t t_start = static_cast<size_t>(thread_idx) * tasks_per_thread;
          size_t t_end = std::min(t_start + tasks_per_thread, total_tasks);

          for (size_t ti = t_start; ti < t_end; ++ti) {
            if (state_ == MemoryState::CANCELLED) {
              LOG(INFO) << "ToHostResize cancelled for model " << model_path_;
              return 0;
            }

            const TensorTask& task = tasks[ti];

            // Locate the starting partition for this tensor's file_offset.
            size_t src_ptn = 0;
            size_t src_off = task.file_offset;
            while (src_ptn < partition_sizes_.size() &&
                   src_off >= partition_sizes_[src_ptn]) {
              src_off -= partition_sizes_[src_ptn];
              src_ptn++;
            }
            if (src_ptn >= partition_sizes_.size()) {
              LOG(ERROR) << "task file_offset=" << task.file_offset
                         << " exceeds total file size";
              return -1;
            }

            // Copy tensor data byte-by-byte across chunk and partition boundaries.
            // Both source (file) and destination (pinned buffer) are traversed
            // in a single pass without revisiting any byte → no data races.
            size_t remaining = task.size;
            size_t dst_abs = task.resize_offset;

            while (remaining > 0) {
              size_t cid = dst_abs / chunk_size;
              size_t coff = dst_abs % chunk_size;
              if (cid >= host_buffers.size()) {
                LOG(ERROR) << "chunk_id=" << cid
                           << " out of range (num_chunks=" << host_buffers.size() << ")";
                return -1;
              }

              // Bytes available in the current destination chunk and source partition.
              size_t avail_dst = chunk_size - coff;
              size_t avail_src = partition_sizes_[src_ptn] - src_off;
              size_t to_read = std::min({remaining, avail_dst, avail_src});

              char* dst_ptr = host_buffers[cid] + coff;
              ssize_t ret =
                  pread(file_descriptors[src_ptn], dst_ptr, to_read, src_off);
              if (ret != static_cast<ssize_t>(to_read)) {
                LOG(ERROR) << "pread failed: expected=" << to_read
                           << " got=" << ret << " errno=" << strerror(errno)
                           << " partition=" << src_ptn << " offset=" << src_off;
                return -1;
              }

              remaining -= to_read;
              dst_abs += to_read;
              src_off += to_read;

              // Advance to next partition if current one is exhausted.
              if (src_off >= partition_sizes_[src_ptn]) {
                src_off = 0;
                src_ptn++;
              }
            }

            completed_tasks.fetch_add(1, std::memory_order_relaxed);
          }
          return 0;
        }));
  }

  bool error = false;
  for (auto& future : futures) {
    if (future.get() != 0) {
      LOG(ERROR) << "ToHostResize: worker thread returned error";
      error = true;
    }
  }

  progress_thread_running.store(false);
  if (progress_thread.joinable()) progress_thread.join();

  for (int fd : file_descriptors) close(fd);

  lock.lock();
  if (error) {
    state_ = MemoryState::INTERRUPTED;
    for (auto& [replica_uuid, gpu_replica] : gpu_replicas_) {
      if (gpu_replica->state_ == MemoryState::LOADING) {
        gpu_replica->state_ = MemoryState::CANCELLED;
        gpu_replica->cv_.notify_all();
      }
      gpu_replica->cv_.wait(lock, [&gpu_replica] {
        return gpu_replica->state_ == MemoryState::LOADED ||
               gpu_replica->state_ == MemoryState::INTERRUPTED;
      });
    }
    pinned_mem_.reset();
    state_ = MemoryState::UNALLOCATED;
    return -1;
  }

  // Enqueue all chunks so that DispatchToGpu can consume them.
  // ToHostResize completes fully before ToGpu is called, so batch-enqueueing
  // here is correct and avoids partially-written chunk notifications.
  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    size_t sz = std::min(chunk_size,
                         model_size_resize_ - chunk_idx * chunk_size);
    host_ptr_vector_->enqueue(chunk_idx, Batch{chunk_idx, sz});
  }

  state_ = MemoryState::LOADED;
  LOG(INFO) << "Finished loading model " << model_path_
            << " from disk with resized layout";
  return 0;
}

int Model::EnsureGpuReplica(const std::string& replica_uuid) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(ERROR) << "Model " << model_path_ << " is not initialized";
    return -1;
  }
  auto replica_it = gpu_replicas_.find(replica_uuid);
  if (replica_it != gpu_replicas_.end()) {
    return 0;
  }
  LOG(INFO) << "Pre-creating replica " << replica_uuid;
  gpu_replicas_.emplace(replica_uuid, std::make_shared<GpuReplica>());
  cv_.notify_all();
  return 0;
}

int Model::ToGpu(
    const std::string& replica_uuid, const MemPtrListMap& device_ptrs,
    const std::unordered_map<int, MemCopyChunkList>& mem_copy_chunks,
    const std::unordered_map<int, MemCopyHandleList>& mem_copy_handles) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(ERROR) << "Model " << model_path_ << " is not initialized";
    return -1;
  }

  GpuReplicaPtr gpu_replica;
  auto replica_it = gpu_replicas_.find(replica_uuid);
  if (replica_it != gpu_replicas_.end()) {
    gpu_replica = replica_it->second;
    if (gpu_replica->state_ != MemoryState::UNINITIALIZED) {
      LOG(ERROR) << "Replica " << replica_uuid << " already exists";
      return -1;
    }
    LOG(INFO) << "Initializing pre-created replica " << replica_uuid;
  } else {
    LOG(INFO) << "Creating replica " << replica_uuid;
    gpu_replicas_.emplace(replica_uuid, std::make_shared<GpuReplica>());
    gpu_replica = gpu_replicas_.at(replica_uuid);
  }
  gpu_replica->gpu_loading_queue_.clear();
  for (const auto& [device_id, _] : device_ptrs) {
    LOG(INFO) << "Creating queue for device " << device_id;
    gpu_replica->gpu_loading_queue_.emplace(device_id,
                                            std::make_shared<GpuReplica::PriorityGpuQueue>());
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
          std::vector<char*> host_char_buffers(host_buffers.begin(),
                                               host_buffers.end());
          while (true) {
            uint64_t task_id = 0;
            if (!gpu_loading_queue->Pop(&task_id)) {
              break;
            }
            if (gpu_replica->state_ == MemoryState::CANCELLED) {
              LOG(INFO) << "Loading from mem for model " << model_path_
                        << " is cancelled";
              return 0;
            }
            std::shared_ptr<GpuReplica::CopyTask> task;
            {
              std::shared_lock<std::shared_mutex> task_lock(
                  gpu_replica->task_mu_);
              auto it = gpu_replica->task_map_.find(task_id);
              if (it == gpu_replica->task_map_.end()) {
                continue;
              }
              task = it->second;
            }

            if (!task) {
              continue;
            }
            int expected = 0;
            if (!task->exec_state_.compare_exchange_strong(expected, 1)) {
              continue;
            }
            const size_t task_bit = gpu_replica->TaskBit(task_id);

            if (task->priority() == CopyPriority::LOW) {
              const size_t reorder_bit = task_bit;
              const bool should_reorder =
                  gpu_replica->reorder_bitmap_.test_and_clear(reorder_bit);
              if (should_reorder) {
                task->exec_state_.store(0);
                gpu_loading_queue->Push(task_id, CopyPriority::LOW);
                continue;
              }
            }

            int ret = CopyTaskToGpu(gpu_replica, task, device_ptr_list,
                                    host_char_buffers);
            if (ret != 0) {
              return ret;
            }

            gpu_replica->completed_bitmap_.test_and_set(task_bit);
            gpu_replica->enqueued_bitmap_.clear(task_bit);
            gpu_replica->enqueued_high_bitmap_.clear(task_bit);
            task->exec_state_.store(2);
            gpu_replica->task_cv_.notify_all();
          }

          LOG(INFO) << "Finished loading tensor from memory to device "
                    << device_id;

          return 0;
        }));
  }

  LOG(INFO) << "Waiting for model " << model_path_ << " num tasks "
            << futures.size() << " state " << gpu_replica->state_;
  int dispatch_ret = dispatch_future.get();
  for (auto& [device_id, gpu_loading_queue] : gpu_replica->gpu_loading_queue_) {
    gpu_loading_queue->Close();
  }
  bool error = dispatch_ret != 0;
  if (dispatch_ret != 0) {
    LOG(ERROR) << "DispatchToGpu failed for model " << model_path_;
  }
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

int Model::SubmitHighPriorityTasks(const std::string& replica_uuid,
                                   const std::vector<uint64_t>& task_ids,
                                   std::vector<uint64_t>* pending_task_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto replica_it = gpu_replicas_.find(replica_uuid);
  if (replica_it == gpu_replicas_.end()) {
    LOG(ERROR) << "Replica " << replica_uuid << " not found";
    return -1;
  }
  auto gpu_replica = replica_it->second;
  lock.unlock();

  std::unordered_set<uint64_t> dedup(task_ids.begin(), task_ids.end());
  for (uint64_t task_id : dedup) {
    auto bit = gpu_replica->TaskBit(task_id);
    if (gpu_replica->completed_bitmap_.test(bit)) {
      continue;
    }

    std::shared_ptr<GpuReplica::CopyTask> task;
    {
      std::unique_lock<std::shared_mutex> task_lock(gpu_replica->task_mu_);
      auto task_it = gpu_replica->task_map_.find(task_id);
      if (task_it == gpu_replica->task_map_.end()) {
        if (pending_task_ids != nullptr) {
          pending_task_ids->push_back(task_id);
        }
        continue;
      }
      task = task_it->second;
      task->set_priority(CopyPriority::HIGH);
    }

    if (task && task->ready_.load(std::memory_order_acquire) &&
        !gpu_replica->enqueued_high_bitmap_.test_and_set(bit)) {
      auto queue_it = gpu_replica->gpu_loading_queue_.find(task->device_id_);
      if (queue_it != gpu_replica->gpu_loading_queue_.end()) {
        queue_it->second->Push(task_id, CopyPriority::HIGH);
      }
    }
  }
  return 0;
}

int Model::SetReorderBitmap(const std::string& replica_uuid,
                            const std::vector<uint64_t>& task_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto replica_it = gpu_replicas_.find(replica_uuid);
  if (replica_it == gpu_replicas_.end()) {
    LOG(ERROR) << "Replica " << replica_uuid << " not found";
    return -1;
  }
  auto gpu_replica = replica_it->second;
  lock.unlock();

  gpu_replica->reorder_bitmap_.reset();
  for (uint64_t task_id : task_ids) {
    auto bit = gpu_replica->TaskBit(task_id);
    if (gpu_replica->completed_bitmap_.test(bit)) {
      continue;
    }
    gpu_replica->reorder_bitmap_.test_and_set(bit);
  }
  return 0;
}

int Model::WaitCopyTasks(const std::string& replica_uuid,
                         const std::vector<uint64_t>& task_ids,
                         uint64_t timeout_ms,
                         std::vector<uint64_t>* pending_task_ids) {
  std::unique_lock<std::mutex> lock(mutex_);
  auto replica_it = gpu_replicas_.find(replica_uuid);
  if (replica_it == gpu_replicas_.end()) {
    LOG(ERROR) << "Replica " << replica_uuid << " not found";
    return -1;
  }
  auto gpu_replica = replica_it->second;
  lock.unlock();

  std::unordered_set<uint64_t> dedup(task_ids.begin(), task_ids.end());
  dedup.reserve(task_ids.size());
  auto collect_pending = [&]() {
    bool all_done = true;
    if (pending_task_ids != nullptr) {
      pending_task_ids->clear();
    }
    for (uint64_t task_id : dedup) {
      if (!gpu_replica->completed_bitmap_.test(gpu_replica->TaskBit(task_id))) {
        all_done = false;
        if (pending_task_ids != nullptr) {
          pending_task_ids->push_back(task_id);
        }
      }
    }
    return all_done;
  };

  if (collect_pending()) {
    return 0;
  }

  std::unique_lock<std::mutex> wait_lock(gpu_replica->wait_mu_);
  bool finished = false;
  if (timeout_ms == 0) {
    finished = gpu_replica->task_cv_.wait_for(
        wait_lock, std::chrono::milliseconds(1), collect_pending);
  } else {
    finished = gpu_replica->task_cv_.wait_for(
        wait_lock, std::chrono::milliseconds(timeout_ms), collect_pending);
  }

  return finished ? 0 : 1;
}

int Model::CopyTaskToGpu(const std::shared_ptr<GpuReplica>& gpu_replica,
                         const std::shared_ptr<GpuReplica::CopyTask>& task,
                         const std::vector<void*>& device_ptr_list,
                         const std::vector<char*>& host_buffers) {
  for (const auto& part : task->parts_) {
    if (part.handle_idx_ >= device_ptr_list.size()) {
      LOG(ERROR) << "Invalid handle index " << part.handle_idx_
                 << " for task " << task->task_id_;
      return -1;
    }
    if (part.chunk_id_ >= host_buffers.size()) {
      LOG(ERROR) << "Invalid chunk id " << part.chunk_id_ << " for task "
                 << task->task_id_;
      return -1;
    }
    CUDA_CHECK(
        cudaMemcpy((void*)((char*)device_ptr_list[part.handle_idx_] +
                           part.gpu_offset_),
                   (void*)(host_buffers[part.chunk_id_] + part.chunk_offset_),
                   part.size_, cudaMemcpyHostToDevice),
        "cudaMemcpy Error");
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
    LOG(WARNING) << "Model " << model_path_ << " replica " << replica_uuid
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
    LOG(WARNING) << "Model " << model_path_ << " is not initialized";
    return 1;
  }

  if (state_ == MemoryState::UNALLOCATED) {
    LOG(WARNING) << "Model " << model_path_ << " is not allocated";
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
    LOG(WARNING) << "Model " << model_path_ << " is not initialized";
    return 0;
  }

  if (state_ == MemoryState::UNALLOCATED) {
    LOG(WARNING) << "Model " << model_path_ << " is not allocated";
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
  if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "CPU memory not allocated";
    return -1;
  }

  size_t num_chunks = pinned_mem_->num_chunks();
  size_t estimated_tasks = 0;
  for (const auto& [_, mem_copy_chunk_list] : mem_copy_chunks) {
    estimated_tasks += mem_copy_chunk_list.size();
  }
  {
    std::unique_lock<std::shared_mutex> lock(gpu_replica->task_mu_);
    gpu_replica->task_map_.reserve(gpu_replica->task_map_.size() +
                                   estimated_tasks);
  }
  std::vector<std::vector<uint64_t>> chunk_id_to_task_ids(num_chunks);
  std::unordered_map<uint64_t, size_t> task_total_parts;
  std::unordered_map<uint64_t, size_t> task_ready_parts;
  task_total_parts.reserve(estimated_tasks);
  task_ready_parts.reserve(estimated_tasks);

  for (const auto& [device_id, mem_copy_chunk_list] : mem_copy_chunks) {
    if (mem_copy_handles.find(device_id) == mem_copy_handles.end()) {
      LOG(ERROR) << "No mem handles for device " << device_id;
      return -1;
    }
    const auto& device_handles = mem_copy_handles.at(device_id);
    std::vector<size_t> handle_offsets(device_handles.size(), 0);

    for (const auto& chunk : mem_copy_chunk_list) {
      const auto host_offset = chunk.src_offset_;
      const auto size = chunk.size_;
      const auto gpu_offset = chunk.dst_offset_;
      const auto handle_idx = chunk.handle_idx_;
      if (handle_idx >= handle_offsets.size()) {
        LOG(ERROR) << "Invalid handle index " << handle_idx << " for device "
                   << device_id;
        return -1;
      }

      uint64_t task_id = chunk.task_id_;
      if (task_id == 0) {
        LOG(ERROR) << "Invalid task_id=0 for device " << device_id
                   << " host_offset=" << host_offset << " size=" << size
                   << " dst_offset=" << gpu_offset
                   << ". Client must provide non-zero task_id.";
        return -1;
      }
      CopyPriority priority =
          chunk.priority_ == static_cast<uint8_t>(CopyPriority::HIGH)
              ? CopyPriority::HIGH
              : CopyPriority::LOW;

      std::shared_ptr<GpuReplica::CopyTask> task;
      {
        std::unique_lock<std::shared_mutex> lock(gpu_replica->task_mu_);
        auto task_it = gpu_replica->task_map_.find(task_id);
        if (task_it == gpu_replica->task_map_.end()) {
          task = std::make_shared<GpuReplica::CopyTask>();
          task->task_id_ = task_id;
          task->device_id_ = device_id;
          task->set_priority(priority);
          task->reorder_hint_ = chunk.reorder_hint_;
          gpu_replica->task_map_[task_id] = task;
        } else {
          task = task_it->second;
          if (priority == CopyPriority::HIGH) {
            task->set_priority(CopyPriority::HIGH);
          }
          if (chunk.reorder_hint_) {
            task->reorder_hint_ = true;
          }
        }
      }

      handle_offsets[handle_idx] = gpu_offset;
      std::vector<std::tuple<int, size_t, size_t>> parts =
          MapDataToChunks(host_offset, size, pinned_mem_->chunk_size());
      std::vector<GpuReplica::TaskPart> new_parts;
      new_parts.reserve(parts.size());
      for (const auto& [chunk_id, chunk_offset, part_size] : parts) {
        GpuReplica::TaskPart task_part;
        task_part.chunk_id_ = static_cast<size_t>(chunk_id);
        task_part.chunk_offset_ = chunk_offset;
        task_part.size_ = part_size;
        task_part.gpu_offset_ = handle_offsets[handle_idx];
        task_part.handle_idx_ = handle_idx;
        new_parts.push_back(task_part);
        task_total_parts[task_id] += 1;
        if (chunk_id >= 0 && static_cast<size_t>(chunk_id) < num_chunks) {
          chunk_id_to_task_ids[chunk_id].push_back(task_id);
        } else {
          LOG(ERROR) << "Chunk id out of range: " << chunk_id;
          return -1;
        }
        handle_offsets[handle_idx] += part_size;
      }
      if (!new_parts.empty()) {
        std::unique_lock<std::shared_mutex> lock(gpu_replica->task_mu_);
        task->parts_.insert(task->parts_.end(), new_parts.begin(),
                            new_parts.end());
      }

      if (chunk.reorder_hint_) {
        gpu_replica->reorder_bitmap_.test_and_set(gpu_replica->TaskBit(task_id));
      }
    }
  }

  /*
  // Original behavior: enqueue as soon as a task becomes ready while scanning
  // chunks, which makes enqueue order depend on chunk ready order.
  for (int i = 0; i < host_ptr_vector_->capacity(); i++) {
    auto data_chunk = host_ptr_vector_->dequeue(i);
    auto chunk_id = data_chunk.chunk_id_;
    auto& task_ids = chunk_id_to_task_ids[chunk_id];
    for (uint64_t task_id : task_ids) {
      task_ready_parts[task_id] += 1;
      if (task_ready_parts[task_id] < task_total_parts[task_id]) {
        continue;
      }
      std::shared_ptr<GpuReplica::CopyTask> task;
      {
        std::shared_lock<std::shared_mutex> lock(gpu_replica->task_mu_);
        auto task_it = gpu_replica->task_map_.find(task_id);
        if (task_it == gpu_replica->task_map_.end()) {
          continue;
        }
        task = task_it->second;
      }
      task->ready_.store(true, std::memory_order_release);

      const size_t bit = gpu_replica->TaskBit(task_id);
      if (gpu_replica->completed_bitmap_.test(bit)) {
        continue;
      }
      if (gpu_replica->enqueued_bitmap_.test_and_set(bit)) {
        continue;
      }
      auto queue_it = gpu_replica->gpu_loading_queue_.find(task->device_id_);
      if (queue_it == gpu_replica->gpu_loading_queue_.end()) {
        LOG(ERROR) << "No queue for device " << task->device_id_;
        return -1;
      }
      queue_it->second->Push(task_id, task->priority());
    }
  }
  */

  // New behavior: task_id-priority enqueue.
  // First mark ready tasks, then enqueue in ascending task_id order.
  std::unordered_set<uint64_t> ready_task_ids;
  ready_task_ids.reserve(task_total_parts.size());
  for (int i = 0; i < host_ptr_vector_->capacity(); i++) {
    auto data_chunk = host_ptr_vector_->dequeue(i);
    auto chunk_id = data_chunk.chunk_id_;
    auto& task_ids = chunk_id_to_task_ids[chunk_id];
    for (uint64_t task_id : task_ids) {
      task_ready_parts[task_id] += 1;
      if (task_ready_parts[task_id] == task_total_parts[task_id]) {
        ready_task_ids.insert(task_id);
      }
    }
  }

  std::vector<uint64_t> ready_task_ids_sorted(ready_task_ids.begin(),
                                              ready_task_ids.end());
  std::sort(ready_task_ids_sorted.begin(), ready_task_ids_sorted.end());
  for (uint64_t task_id : ready_task_ids_sorted) {
    std::shared_ptr<GpuReplica::CopyTask> task;
    {
      std::shared_lock<std::shared_mutex> lock(gpu_replica->task_mu_);
      auto task_it = gpu_replica->task_map_.find(task_id);
      if (task_it == gpu_replica->task_map_.end()) {
        continue;
      }
      task = task_it->second;
    }
    task->ready_.store(true, std::memory_order_release);

    const size_t bit = gpu_replica->TaskBit(task_id);
    if (gpu_replica->completed_bitmap_.test(bit)) {
      continue;
    }
    if (gpu_replica->enqueued_bitmap_.test_and_set(bit)) {
      continue;
    }
    auto queue_it = gpu_replica->gpu_loading_queue_.find(task->device_id_);
    if (queue_it == gpu_replica->gpu_loading_queue_.end()) {
      LOG(ERROR) << "No queue for device " << task->device_id_;
      return -1;
    }
    queue_it->second->Push(task_id, task->priority());
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
  pinned_mem_ = std::make_shared<PinnedMemory>();
  int ret = pinned_mem_->Allocate(model_size_, pool);
  if (ret < 0) {
    LOG(ERROR) << "Error allocating CPU memory for model " << model_path_;
    return ret;
  } else if (ret > 0) {
    LOG(WARNING) << "Not enough memory for model " << model_path_;
    return ret;
  } else if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "CPU memory not allocated";
    return -1;
  }

  state_ = MemoryState::ALLOCATED;
  return 0;
};

int Model::AllocatePinnedMemory(std::shared_ptr<PinnedMemoryPoolShared> pool) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == MemoryState::UNINITIALIZED) {
    LOG(ERROR) << "Model " << model_path_ << " is not initialized";
    return -1;
  }
  if (state_ != MemoryState::UNALLOCATED) {
    return 0;
  }
  pinned_mem_ = std::make_shared<PinnedMemory>();
  int ret = pinned_mem_->Allocate(model_size_, pool);
  if (ret < 0) {
    LOG(ERROR) << "Error allocating shared CPU memory for model " << model_path_;
    return ret;
  } else if (ret > 0) {
    LOG(WARNING) << "Not enough shared memory for model " << model_path_;
    return ret;
  } else if (!pinned_mem_ || pinned_mem_->num_chunks() == 0) {
    LOG(ERROR) << "Shared CPU memory not allocated";
    return -1;
  }

  state_ = MemoryState::ALLOCATED;
  return 0;
}