// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/vendors/intel_openvino/compiler/weight_bank.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "litert/c/internal/litert_logging.h"
#include "litert/compiler/cc/litert_model.h"

namespace litert::openvino {

void WeightBank::AddSubgraph(const litert::compiler::Subgraph& subgraph) {
  for (const auto& op : subgraph.Ops()) {
    for (const auto& input : op.Inputs()) {
      if (!input.HasWeights()) {
        continue;
      }
      const auto weights = input.Weights();
      // Keyed by BufferId, so a buffer shared by multiple ops/partitions is
      // recorded once. The bytes are identical for a given id, so re-assignment
      // is harmless.
      const int32_t buffer_id = weights.BufferId();
      // Defensive: BufferId() returns -1 only when the id lookup fails (missing
      // callback / bad handle). A real weight always has a valid id, so skip
      // rather than pollute the pool with a sentinel key.
      if (buffer_id < 0) {
        continue;
      }
      buffer_bytes_[buffer_id] = weights.Bytes();
      // Record this tensor's name so the matching OpenVINO weight (which takes
      // the tensor name as its friendly_name) can be resolved back to its
      // buffer. Distinct names sharing a buffer all point at the same id.
      name_to_buffer_id_[std::string(input.Name())] = buffer_id;
    }
  }
}

size_t WeightBank::TotalBytes() const {
  size_t total = 0;
  for (const auto& [buffer_id, bytes] : buffer_bytes_) {
    total += bytes.size();
  }
  return total;
}

int32_t WeightBank::RegisterOrGetDerivedBuffer(std::string_view key,
                                               std::vector<uint8_t> bytes,
                                               int partition_idx) {
  const std::string key_str(key);
  if (auto it = derived_key_to_buffer_id_.find(key_str);
      it != derived_key_to_buffer_id_.end()) {
    const int owner_partition = derived_key_to_owner_partition_.at(key_str);
    if (owner_partition == partition_idx) {
      // Same partition re-hitting a key it already owns: this is two
      // logically DIFFERENT constants (e.g. two different layers' norm-gain)
      // that merely hash the same within this one partition, not a
      // legitimate cross-partition match. Mint an independent BufferId
      // instead of merging -- and deliberately do NOT record it under |key|,
      // so it can never be (mis)matched again later either.
      const int32_t buffer_id =
          kDerivedBufferIdBase + static_cast<int32_t>(derived_storage_.size());
      derived_storage_.push_back(std::move(bytes));
      buffer_bytes_[buffer_id] = absl::MakeConstSpan(derived_storage_.back());
      LITERT_LOG(LITERT_INFO,
                 "WeightBank: derived buffer key '%s' seen again within "
                 "partition %d (its own owner); treating as an independent, "
                 "unshared buffer instead of merging",
                 key_str.c_str(), partition_idx);
      return buffer_id;
    }
    // A key match from a DIFFERENT partition is a legitimate cross-partition
    // match (that's the point of hashing content). This is just a loud
    // tripwire in case that assumption is ever wrong (hash collision).
    // AliasAndTagSharedConstants' own byte-compare still protects against a
    // collision silently corrupting the shared pool, so this cannot itself
    // cause a correctness bug -- it only makes an otherwise-silent
    // near-impossible event visible.
    const absl::Span<const uint8_t> existing = buffer_bytes_.at(it->second);
    if (existing.size() != bytes.size() ||
        std::memcmp(existing.data(), bytes.data(), bytes.size()) != 0) {
      LITERT_LOG(LITERT_ERROR,
                 "WeightBank: derived buffer key '%s' hash-collided with "
                 "different content (%zu vs %zu bytes); keeping the first "
                 "registration",
                 key_str.c_str(), existing.size(), bytes.size());
    }
    return it->second;
  }
  const int32_t buffer_id =
      kDerivedBufferIdBase + static_cast<int32_t>(derived_storage_.size());
  derived_storage_.push_back(std::move(bytes));
  buffer_bytes_[buffer_id] = absl::MakeConstSpan(derived_storage_.back());
  derived_key_to_buffer_id_[key_str] = buffer_id;
  derived_key_to_owner_partition_[key_str] = partition_idx;
  return buffer_id;
}

std::optional<int32_t> WeightBank::BufferIdOfName(
    std::string_view tensor_name) const {
  auto name_it = name_to_buffer_id_.find(std::string(tensor_name));
  if (name_it == name_to_buffer_id_.end()) {
    return std::nullopt;
  }
  return name_it->second;
}

}  // namespace litert::openvino
