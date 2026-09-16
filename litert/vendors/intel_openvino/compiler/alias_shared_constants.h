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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_set>
#include <vector>

#include "absl/types/span.h"  // from @com_google_absl
#include "litert/vendors/intel_openvino/compiler/weight_bank.h"
#include "openvino/core/model.hpp"

namespace litert::openvino {

// Resolves and stamps LiteRtBufferIdAttribute identity (see
// buffer_id_attribute.h) onto every sharing-candidate Constant (>=16
// elements) in |ov_model|. Call this ONCE per partition, right after
// OpenVinoCompileContext::OptimizeModel() returns -- this is the only place
// pool identity is decided; nothing upstream of it (frontend conversion, the
// NPU optimizer passes) needs to know about WeightBank at all.
//
// Two ways a Constant resolves:
//   - Its friendly_name matches an original LiteRt weight tensor recorded by
//     weight_bank.AddSubgraph() (weight_bank.BufferIdOfName()): it survived
//     optimization untouched.
//   - Otherwise: it is a NEW Constant synthesized by some optimizer pass
//     (e.g. MoE expert-weight stacking) with no LiteRt name lineage. Its
//     identity can only be recovered from its own byte content, so its bytes
//     are hashed and registered/deduped via
//     weight_bank.RegisterOrGetDerivedBuffer() -- which only merges a hash
//     match across DIFFERENT partitions (a legitimate shared tensor); a
//     match against this SAME |partition_idx| gets its own independent id
//     instead (two logically different constants, e.g. two different
//     layers' norm-gain, that merely hash the same within one partition).
//
// Constants below the 16-element threshold (shape/control constants) are left
// untouched -- never worth pooling.
void HarvestSharedConstants(const std::shared_ptr<ov::Model>& ov_model,
                            WeightBank& weight_bank, int partition_idx);

// Returns the BufferId of every Constant in |ov_model| carrying a
// LiteRtBufferIdAttribute (i.e. every Constant HarvestSharedConstants
// resolved) -- exactly the set of ids AliasAndTagSharedConstants(ov_model,
// ...) would go on to alias/tag, for whatever pool_offset_of it is eventually
// given.
//
// Call this AFTER HarvestSharedConstants has run on every partition, to
// decide which buffers are still referenced by SOME partition and which are
// dead -- e.g. an MoE layer's disaggregated per-expert Constants, once every
// partition that used them has replaced them with a stacked derived
// Constant, become unreachable from any Result and so will not appear here,
// while a partition whose rewrite didn't apply for that layer still has them
// reachable and keeps them counted as referenced.
//
// Uses the exact same Constant-candidate resolution as
// AliasAndTagSharedConstants (see its implementation) so the two can never
// disagree on what counts as "still referenced" -- there is only one
// implementation of that resolution, shared by both call sites.
std::unordered_set<int32_t> CollectReferencedBufferIds(
    const std::shared_ptr<ov::Model>& ov_model);

// One buffer's placement in the assembled cross-partition pool.
struct PoolEntry {
  int32_t buffer_id = 0;
  size_t pool_offset = 0;  // byte offset within the contiguous pool
  absl::Span<const uint8_t> bytes;
};

// The final pool layout: buffers in ascending BufferId order (the layout
// OpenVinoGlobalGraph::Serialize() requires), plus the BufferId -> pool_offset
// map AliasAndTagSharedConstants needs.
struct PoolLayout {
  std::vector<PoolEntry> buffers;
  std::map<int32_t, size_t> pool_offset_of;
};

// Assembles the final cross-partition shared pool from |weight_bank|'s
// recorded buffers, in ascending BufferId order. When |prune_dead| is true, a
// buffer is kept only if at least one model in |ov_models| still references it
// (per CollectReferencedBufferIds) -- e.g. an MoE layer's disaggregated
// per-expert buffers, once every partition has replaced them with a stacked
// derived Constant, are dropped. This is a UNION across |ov_models|, NOT a
// "referenced by more than one partition" filter: a buffer used by only one
// partition is kept exactly like one shared by all of them, as long as SOME
// partition still references it. When |prune_dead| is false (GPU), every
// recorded buffer is kept unconditionally.
PoolLayout BuildPool(const std::vector<std::shared_ptr<ov::Model>>& ov_models,
                     const WeightBank& weight_bank, bool prune_dead);

// NPU cross-partition weight-sharing transform (counterpart to the GPU
// ConvertWeightsToParameters).
//
// Rebuilds each large bank-backed weight Constant in |ov_model| so its data
// pointer ALIASES the deduplicated pool bytes for its BufferId (one stable host
// pointer shared across every partition), then stamps its
// WeightlessCacheAttribute(bin_offset).
//
// Aliasing is the load-bearing step for NPU weight sharing: NPUW dedups weights
// by their Constant data pointer, so pointing every partition's Constant at the
// one pool buffer is what makes NPUW collapse them to a single allocation.
// bin_offset tells NPUW where to mmap the weight at runtime.
//
// A Constant is aliased ONLY when its current bytes are byte-identical to the
// pool bytes for its BufferId. A mismatch means a content-altering frontend
// transform rewrote this weight, so aliasing would feed the graph the wrong
// data; such weights are left baked/per-partition. Comparing bytes is the
// robust guard -- no need to enumerate which transforms alter content.
//
// |pool_offset_of| maps BufferId -> byte offset in the contiguous shared pool
// (the same ascending-id layout Serialize() uses). |partition_idx| is used only
// for logging.
//
// Returns the number of Constants aliased (i.e. actually shareable). Constants
// with no BufferId or below the element threshold stay baked.
size_t AliasAndTagSharedConstants(
    const std::shared_ptr<ov::Model>& ov_model, const WeightBank& weight_bank,
    const std::map<int32_t, size_t>& pool_offset_of, int partition_idx);

}  // namespace litert::openvino

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_ALIAS_SHARED_CONSTANTS_H_
