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
#include "pinned_memory_pool_shared.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <sstream>

PinnedMemoryPoolShared::PinnedMemoryPoolShared(size_t total_size,
                                               size_t chunk_size,
                                               const std::string& shm_name_prefix)
    : chunk_size_(chunk_size),
      buffer_counter_(0),
      shm_name_prefix_(shm_name_prefix) {
  size_t num_buffers = (total_size + chunk_size - 1) / chunk_size;
  if (num_buffers * chunk_size != total_size) {
    LOG(ERROR) << "PinnedMemoryPoolShared size not multiple of chunk_size";
  }
  LOG(INFO) << "Creating PinnedMemoryPoolShared with " << num_buffers
            << " buffers of " << chunk_size << " bytes, prefix: "
            << shm_name_prefix_;

  pid_t pid = getpid();
  for (size_t i = 0; i < num_buffers; ++i) {
    // Generate unique shared memory name
    std::ostringstream oss;
    oss << shm_name_prefix_ << "_" << pid << "_" << buffer_counter_++;
    std::string shm_name = oss.str();

    // Create shared memory object
    int shm_fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shm_fd == -1) {
      LOG(FATAL) << "shm_open failed for " << shm_name << ": "
                 << strerror(errno);
    }

    // Set the size of the shared memory object
    if (ftruncate(shm_fd, chunk_size_) == -1) {
      close(shm_fd);
      shm_unlink(shm_name.c_str());
      LOG(FATAL) << "ftruncate failed for " << shm_name << ": "
                 << strerror(errno);
    }

    // Map shared memory to process address space with 4096-byte alignment
    // MAP_SHARED makes it accessible by other processes
    void* mapped_addr = mmap(nullptr, chunk_size_, PROT_READ | PROT_WRITE,
                             MAP_SHARED, shm_fd, 0);
    close(shm_fd);  // Close file descriptor, mmap keeps the mapping

    if (mapped_addr == MAP_FAILED) {
      shm_unlink(shm_name.c_str());
      LOG(FATAL) << "mmap failed for " << shm_name << ": " << strerror(errno);
    }

    // Ensure 4096-byte alignment (mmap should already provide page alignment)
    char* buffer = static_cast<char*>(mapped_addr);
    if (reinterpret_cast<uintptr_t>(buffer) % 4096 != 0) {
      munmap(mapped_addr, chunk_size_);
      shm_unlink(shm_name.c_str());
      LOG(FATAL) << "Shared memory not aligned to 4096 bytes for "
                 << shm_name;
    }

    // Register with CUDA as pinned memory
    // This makes the memory accessible to CUDA and allows fast GPU transfers
    cudaError_t err =
        cudaHostRegister(buffer, chunk_size_, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
      munmap(buffer, chunk_size_);
      shm_unlink(shm_name.c_str());
      LOG(FATAL) << "cudaHostRegister failed for " << shm_name << ": "
                 << cudaGetErrorString(err);
    }

    pool_.insert(buffer);
    free_list_.insert(buffer);
    shm_names_[buffer] = shm_name;

    LOG(INFO) << "Created shared pinned memory buffer " << shm_name
              << " at address " << static_cast<void*>(buffer);
  }
}

PinnedMemoryPoolShared::~PinnedMemoryPoolShared() {
  for (char* buffer : pool_) {
    cudaHostUnregister(buffer);
    munmap(buffer, chunk_size_);
    // Unlink shared memory object
    auto it = shm_names_.find(buffer);
    if (it != shm_names_.end()) {
      shm_unlink(it->second.c_str());
      LOG(INFO) << "Unlinked shared memory " << it->second;
    }
  }
}

int PinnedMemoryPoolShared::Allocate(size_t size,
                                      std::vector<char*>& buffers,
                                      std::vector<std::string>& shm_names) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (size == 0) {
    LOG(ERROR) << "PinnedMemoryPoolShared Allocate size is zero";
    return -1;
  }

  int num_buffers_needed = (size + chunk_size_ - 1) / chunk_size_;
  if (num_buffers_needed > free_list_.size()) {
    LOG(ERROR) << "PinnedMemoryPoolShared out of memory ("
               << free_list_.size() << " buffers available, "
               << num_buffers_needed << " buffers needed)";
    return num_buffers_needed - free_list_.size();
  }

  buffers.clear();
  buffers.resize(num_buffers_needed);
  shm_names.clear();
  shm_names.resize(num_buffers_needed);
  
  auto it = free_list_.begin();
  for (size_t i = 0; i < num_buffers_needed; ++i) {
    char* buffer = *it;
    buffers[i] = buffer;
    
    // Get the shared memory name for this buffer
    auto shm_it = shm_names_.find(buffer);
    if (shm_it != shm_names_.end()) {
      shm_names[i] = shm_it->second;
      LOG(INFO) << "shm_it: " << shm_it->second << " buffer: " << i;
    } else {
      LOG(ERROR) << "Shared memory name not found for buffer "
                 << static_cast<void*>(buffer);
      shm_names[i] = "";
    }
    
    it = free_list_.erase(it);
  }

  LOG(INFO) << "PinnedMemoryPoolShared Allocate " << buffers.size()
            << " buffers, free buffers " << free_list_.size()
            << " total buffers " << pool_.size();

  return 0;  // Success
}

int PinnedMemoryPoolShared::Deallocate(std::vector<char*>& buffers) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (char* buffer : buffers) {
    if (pool_.find(buffer) == pool_.end()) {
      LOG(ERROR) << "Buffer not found in pool";
      return -1;
    }
    if (free_list_.find(buffer) != free_list_.end()) {
      LOG(ERROR) << "Buffer already in free list";
      return -1;
    }
    free_list_.insert(buffer);
  }
  LOG(INFO) << "Deallocated " << buffers.size() << " buffers";
  return 0;  // Success
}

std::string PinnedMemoryPoolShared::GetSharedMemoryName(char* buffer) const {
  auto it = shm_names_.find(buffer);
  if (it != shm_names_.end()) {
    return it->second;
  }
  return "";
}

char* PinnedMemoryPoolShared::AttachToSharedMemory(const std::string& shm_name,
                                                    size_t size) {
  // Open existing shared memory object
  int shm_fd = shm_open(shm_name.c_str(), O_RDWR, 0);
  if (shm_fd == -1) {
    LOG(ERROR) << "Failed to open shared memory " << shm_name << ": "
               << strerror(errno);
    return nullptr;
  }

  // Map shared memory to process address space
  void* mapped_addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                           shm_fd, 0);
  close(shm_fd);  // Close file descriptor, mmap keeps the mapping

  if (mapped_addr == MAP_FAILED) {
    LOG(ERROR) << "Failed to mmap shared memory " << shm_name << ": "
               << strerror(errno);
    return nullptr;
  }

  // Register with CUDA as pinned memory
  cudaError_t err =
      cudaHostRegister(mapped_addr, size, cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    munmap(mapped_addr, size);
    LOG(ERROR) << "Failed to register shared memory with CUDA " << shm_name
               << ": " << cudaGetErrorString(err);
    return nullptr;
  }

  LOG(INFO) << "Attached to shared memory " << shm_name << " at address "
            << mapped_addr;
  return static_cast<char*>(mapped_addr);
}

void PinnedMemoryPoolShared::DetachFromSharedMemory(char* buffer, size_t size) {
  if (buffer == nullptr) {
    return;
  }

  cudaHostUnregister(buffer);
  munmap(buffer, size);
  LOG(INFO) << "Detached from shared memory at address "
            << static_cast<void*>(buffer);
}

