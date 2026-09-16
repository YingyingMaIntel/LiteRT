// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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

#include "litert/vendors/intel_openvino/compiler/alias_shared_constants.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/internal/litert_logging.h"
#include "litert/vendors/intel_openvino/compiler/buffer_id_attribute.h"
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "litert/vendors/intel_openvino/compiler/weightless_caching_attributes.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"

namespace litert::openvino {

namespace {

std::vector<std::pair<std::shared_ptr<ov::op::v0::Constant>, int32_t>>
ResolveSharedConstants(const std::shared_ptr<ov::Model>& ov_model) {
  std::vector<std::pair<std::shared_ptr<ov::op::v0::Constant>, int32_t>> resolved;
  // Collect before any caller mutates the graph: replacing nodes while iterating
  // get_ordered_ops() is unsafe.
  for (const auto& node : ov_model->get_ordered_ops()) {
    auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!cnst) continue;
    const auto& rt = cnst->get_rt_info();
    const auto it = rt.find(LiteRtBufferIdAttribute::get_type_info_static());
    if (it == rt.end())
      continue;  // never harvested: not backed by a shared buffer
    resolved.emplace_back(cnst,
                          it->second.as<LiteRtBufferIdAttribute>().buffer_id);
  }
  return resolved;
}

// FNV-1a running hash, so large buffers can be hashed without an intermediate
// copy into a std::string.
uint64_t FnvHashBytes(const void* data, size_t size, uint64_t hash) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= bytes[i];
    hash *= /*FNV prime=*/1099511628211ULL;
  }
  return hash;
}

}  // namespace

void HarvestSharedConstants(const std::shared_ptr<ov::Model>& ov_model,
                            WeightBank& weight_bank, int partition_idx) {
  size_t by_name = 0;
  size_t by_hash = 0;
  for (const auto& node : ov_model->get_ordered_ops()) {
    auto cnst = ov::as_type_ptr<ov::op::v0::Constant>(node);
    if (!cnst) continue;
    const size_t elem_size = cnst->get_element_type().size();
    // Skip tiny/shape/scalar constants (fewer than 16 elements) and any type
    // with unknown element size -- never worth pooling.
    if (elem_size == 0 || cnst->get_byte_size() / elem_size < 16) continue;

    auto& rt = cnst->get_rt_info();
    if (const auto bid =
            weight_bank.BufferIdOfName(cnst->get_friendly_name())) {
      // Its name still matches an original LiteRt weight tensor: nothing
      // touched this Constant since conversion.
      rt[LiteRtBufferIdAttribute::get_type_info_static()] =
          LiteRtBufferIdAttribute(*bid);
      ++by_name;
      continue;
    }

    // No name lineage: some optimizer pass synthesized this Constant (e.g.
    // MoE expert-weight stacking). Its only stable identity is its own byte
    // content.
    constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
    const uint64_t hash = FnvHashBytes(cnst->get_data_ptr(),
                                       cnst->get_byte_size(), kFnvOffsetBasis);
    std::ostringstream key;
    key << "Derived:" << std::hex << hash;
    std::vector<uint8_t> bytes(cnst->get_byte_size());
    std::memcpy(bytes.data(), cnst->get_data_ptr(), bytes.size());
    const int32_t bid =
        weight_bank.RegisterOrGetDerivedBuffer(key.str(), std::move(bytes),
                                              partition_idx);
    rt[LiteRtBufferIdAttribute::get_type_info_static()] =
        LiteRtBufferIdAttribute(bid);
    ++by_hash;
  }
  LITERT_LOG(LITERT_INFO,
             "HarvestSharedConstants: %zu resolved by name, %zu by content "
             "hash",
             by_name, by_hash);
}

std::unordered_set<int32_t> CollectReferencedBufferIds(
    const std::shared_ptr<ov::Model>& ov_model) {
  std::unordered_set<int32_t> ids;
  for (const auto& [cnst, bid] : ResolveSharedConstants(ov_model)) {
    ids.insert(bid);
  }
  return ids;
}

PoolLayout BuildPool(const std::vector<std::shared_ptr<ov::Model>>& ov_models,
                     const WeightBank& weight_bank, bool prune_dead) {
  std::unordered_set<int32_t> live_buffer_ids;
  if (prune_dead) {
    for (const auto& m : ov_models) {
      for (int32_t id : CollectReferencedBufferIds(m)) {
        live_buffer_ids.insert(id);
      }
    }
  }

  // Ordered map keyed by BufferId so the pool is laid out in ascending id
  // order (matching OpenVinoGlobalGraph::Serialize()).
  std::map<uint32_t, absl::Span<const uint8_t>> ordered(
      weight_bank.Buffers().begin(), weight_bank.Buffers().end());
  PoolLayout layout;
  size_t running_offset = 0;
  size_t pruned = 0, pruned_bytes = 0;
  for (const auto& [buffer_id, bytes] : ordered) {
    if (prune_dead && !live_buffer_ids.count(static_cast<int32_t>(buffer_id))) {
      ++pruned;
      pruned_bytes += bytes.size();
      continue;
    }
    layout.buffers.push_back(
        {static_cast<int32_t>(buffer_id), running_offset, bytes});
    layout.pool_offset_of[static_cast<int32_t>(buffer_id)] = running_offset;
    running_offset += bytes.size();
  }
  if (pruned > 0) {
    LITERT_LOG(LITERT_INFO,
               "Weight sharing: pruned %zu buffer(s) (%zu bytes) superseded "
               "by a stacked/derived constant in every partition that "
               "referenced them",
               pruned, pruned_bytes);
  }
  return layout;
}

size_t AliasAndTagSharedConstants(
    const std::shared_ptr<ov::Model>& ov_model, const WeightBank& weight_bank,
    const std::map<int32_t, size_t>& pool_offset_of, int partition_idx) {
  const auto& buffers = weight_bank.Buffers();
  size_t aliased = 0;
  size_t tagged = 0;
  size_t mismatched = 0;
  for (const auto& [cnst, bid_val] : ResolveSharedConstants(ov_model)) {
    const int32_t bid = bid_val;
    const auto off_it = pool_offset_of.find(bid);
    if (off_it == pool_offset_of.end()) continue;  // not in the shared pool
    const auto buf_it = buffers.find(bid);
    if (buf_it == buffers.end()) continue;  // defensive: id present in map only
    const absl::Span<const uint8_t> pool_bytes = buf_it->second;

    const void* frontend_ptr = cnst->get_data_ptr();
    const size_t cnst_bytes = cnst->get_byte_size();

    // Alias only if the Constant's bytes match the pool bytes (see header
    // comment). Compare size first, then contents.
    const bool bytes_match =
        cnst_bytes == pool_bytes.size() &&
        std::memcmp(frontend_ptr, pool_bytes.data(), cnst_bytes) == 0;

    std::shared_ptr<ov::op::v0::Constant> tag_target = cnst;
    if (bytes_match) {
      // Non-owning Constant over the shared pool bytes (no copy). The pool span
      // views the ONE LiteRt weight mmap, which outlives this compile, so a
      // null keep-alive is safe. get_data_ptr() now returns the pool pointer,
      // so every partition's Constant for this BufferId shares one address ->
      // NPUW dedups.
      auto aliased_cnst = std::make_shared<ov::op::v0::Constant>(
          cnst->get_element_type(), cnst->get_shape(), pool_bytes.data(),
          std::shared_ptr<void>{});
      aliased_cnst->set_friendly_name(cnst->get_friendly_name());
      aliased_cnst->get_output_tensor(0).set_names(
          cnst->get_output_tensor(0).get_names());
      ov::replace_node(cnst, aliased_cnst);
      tag_target = std::move(aliased_cnst);
      ++aliased;
    } else {
      ++mismatched;
    }

    // Stamp the WLCA (bin_offset) on whichever Constant now lives in the graph.
    auto& rt = tag_target->get_rt_info();
    if (!rt.count(ov::WeightlessCacheAttribute::get_type_info_static())) {
      rt[ov::WeightlessCacheAttribute::get_type_info_static()] =
          ov::WeightlessCacheAttribute(tag_target->get_byte_size(),
                                       off_it->second,
                                       tag_target->get_element_type());
      ++tagged;
    }
  }

  LITERT_LOG(
      LITERT_INFO,
      "Weight sharing (NPU) p%d: aliased %zu constants to the shared "
      "pool, tagged %zu with WeightlessCacheAttribute, left %zu unshared "
      "(bytes differ from pool -- content-altered, e.g. i2->u2)",
      partition_idx, aliased, tagged, mismatched);
  return aliased;
}

}  // namespace litert::openvino
