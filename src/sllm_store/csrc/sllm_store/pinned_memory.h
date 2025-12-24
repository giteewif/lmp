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

#include <memory>
#include <string>
#include <vector>

#include "pinned_memory_pool.h"
#include "pinned_memory_pool_shared.h"

class PinnedMemory {
 public:
  PinnedMemory() = default;
  ~PinnedMemory();

  // Disable copying and moving
  PinnedMemory(const PinnedMemory&) = delete;
  PinnedMemory& operator=(const PinnedMemory&) = delete;
  PinnedMemory(PinnedMemory&&) = delete;
  PinnedMemory& operator=(PinnedMemory&&) = delete;

  // Allocate from regular pinned memory pool
  int Allocate(size_t size, std::shared_ptr<PinnedMemoryPool> mempool);
  
  // Allocate from shared pinned memory pool
  int Allocate(size_t size, std::shared_ptr<PinnedMemoryPoolShared> mempool);
  
  std::vector<char*>& get();
  const std::vector<std::string>& get_shm_names() const { return shm_names_; }
  size_t num_chunks() const { return buffers_.size(); }
  size_t chunk_size() const;

 private:
  std::vector<char*> buffers_;
  std::vector<std::string> shm_names_;  // Shared memory names (only used with PinnedMemoryPoolShared)
  std::shared_ptr<PinnedMemoryPool> mempool_;
  std::shared_ptr<PinnedMemoryPoolShared> mempool_shared_;
};
