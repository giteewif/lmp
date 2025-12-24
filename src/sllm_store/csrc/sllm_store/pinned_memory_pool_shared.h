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

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class PinnedMemoryPoolShared {
 public:
  // Constructor: creates shared pinned memory pool
  // shm_name_prefix: prefix for shared memory object names (e.g., "/sllm_pool")
  PinnedMemoryPoolShared(size_t total_size, size_t chunk_size,
                         const std::string& shm_name_prefix = "/sllm_pinned_pool");
  ~PinnedMemoryPoolShared();

  int Allocate(size_t size, std::vector<char*>& buffers,
               std::vector<std::string>& shm_names);
  int Deallocate(std::vector<char*>& buffers);
  size_t chunk_size() const { return chunk_size_; }

  // Get shared memory name for a buffer (for other processes to attach)
  std::string GetSharedMemoryName(char* buffer) const;

  // Static method: attach to existing shared memory by name
  // Returns nullptr on failure
  static char* AttachToSharedMemory(const std::string& shm_name, size_t size);

  // Static method: detach from shared memory
  static void DetachFromSharedMemory(char* buffer, size_t size);

  // Forbid copy and assignment
  PinnedMemoryPoolShared(const PinnedMemoryPoolShared&) = delete;
  PinnedMemoryPoolShared& operator=(const PinnedMemoryPoolShared&) = delete;

 private:
  std::mutex mutex_;
  std::unordered_set<char*> free_list_;
  std::unordered_set<char*> pool_;
  // Map from buffer pointer to shared memory name for cleanup
  std::unordered_map<char*, std::string> shm_names_;
  size_t chunk_size_;
  size_t buffer_counter_;  // For generating unique shared memory names
  std::string shm_name_prefix_;
};

