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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "openvino/core/model.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace litert {
namespace openvino {

// Eliminates FakeQuantize nodes that immediately follow MatMul operations.
//
// The FakeQuantize directly after a MatMul re-quantize its output to a
// calibrated range. This pass detects the MatMul -> FakeQuantize pattern
// and removes the FakeQuantize, rewiring downstream consumers to read
// directly from the MatMul output to reduce NPU overhead.
class EliminateMatMulFakeQuantize : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("EliminateMatMulFakeQuantize");
  EliminateMatMulFakeQuantize();
};

// The Intel NPU compiler's IE.Sign op only accepts float operands
// (f16/f32/f64). When the TFLite frontend produces a Sign node with integer
// input, NPU compilation fails. This pass wraps such Sign nodes with Convert
// ops:
//   Convert(int->f32) -> Sign -> Convert(f32->original_int_type)
class CastIntegerSignToFloat : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("CastIntegerSignToFloat");
  CastIntegerSignToFloat();
};

// Fuses the "split-attention" sub-graph produced by some Gemma-style models
// (attention computed separately against the persistent KV cache and the
// current step's KV, merged via Concat) into a single
// `ov::op::v13::ScaledDotProductAttention`.
//
// Pattern matched (Q is shared between both qk MatMuls):
//
//   qk_cache = MatMul(Q, K_cache)
//   qk_new   = MatMul(Q, K_new)
//   scores   = Concat(qk_cache, qk_new, axis=-1)
//   masked   = Add(scores, mask)
//   probs    = Softmax(masked, axis=-1)
//   pc       = Slice/StridedSlice(probs)        // cache half
//   pn       = Slice/StridedSlice(probs)        // new half
//   attn_c   = MatMul(pc, V_cache)
//   attn_n   = MatMul(pn, V_new)
//   output   = Add(attn_c, attn_n)              // <-- match root
//
// Restricted to static 4-D ranks; other shapes are left untouched. Scale is
// set to 1.0 because the matched pattern has no explicit scale node (any
// scaling is assumed pre-applied to Q by the model, e.g. Gemma's
// `query_pre_attn_scalar`). When the MatMul transpose_b flags imply K or V
// is stored as [B,H,D,S], an explicit Transpose(0,1,3,2) is inserted to
// restore the [B,H,S,D] layout required by v13::SDPA.
//
// Ported from openvinotoolkit/openvino PR #36153.
class FuseSplitAttentionToSDPA : public ov::pass::MatcherPass {
 public:
  OPENVINO_MATCHER_PASS_RTTI("FuseSplitAttentionToSDPA");
  // |pad_kv_to_alignment| controls behavior when the merged KV sequence length
  // is not a multiple of the NPU SDPA kernel's required alignment (16). When
  // false, such blocks are left unfused. When true(default), the merged K/V
  // is padded up to the next multiple of 16 and the mask is padded with a
  // large negative bias so the padded key positions are excluded from softmax.
  explicit FuseSplitAttentionToSDPA(bool pad_kv_to_alignment);
};

// Gives every consumer of a multi-consumer Constant its own private copy of
// that Constant, instead of leaving them sharing one node object.
//
// The TFLite frontend (or the model itself) sometimes hands us a graph where
// two logically-independent tensors that happen to hold identical bytes
// (e.g. two different transformer layers' RMSNorm gain vectors) have ALREADY
// been unified into one literal ov::Constant node with multiple consumers --
// this happens upstream of anything in this compiler, so HarvestSharedConstants
// (alias_shared_constants.cc) merely inherits it. A single shared node across
// what should be independent per-layer slots corrupts anything downstream
// that counts per-layer occurrences to detect a repeated pattern (e.g.
// NPUW's own repeated-subgraph/fold matcher: one layer's constant appears to
// "vanish" because its consumer now points at another layer's node).
//
// Splitting eagerly, before any other pass runs, guarantees every consumer
// sees an independent Constant matching the original per-layer semantics.
// Every multi-consumer Constant is split, regardless of size -- this is
// unrelated to (and safe alongside) the EXPLICIT cross-partition
// weight-sharing this compiler adds later via
// WeightBank/HarvestSharedConstants/AliasAndTagSharedConstants, which never
// depends on the frontend graph having pre-existing shared node identity.
class SplitSharedConstants : public ov::pass::ModelPass {
 public:
  OPENVINO_RTTI("SplitSharedConstants");
  explicit SplitSharedConstants();
  bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

};

// Rewrites Gemma4's dense Mixture-of-Experts block — where all N experts are
// computed and non-selected ones are masked to zero — into a cheaper form.
//
// Gemma4 exports each expert as an independent GEGLU branch with its own
// weights (there is NO batched expert dimension to gather over), the
// branches accumulated via a chain of Add and masked by per-expert router
// scores (Equal(topk_indices, expert_id) -> score, 0 when not selected). Per
// MoE layer this pass locates the router TopK and its N (=128) expert
// branches (expert_id read from each branch's Equal constant), then
// dispatches by the router's token/chunk batch shape:
//   - single-token chunk (TopK indices batch statically == 1): only K
//     experts can be active per call, so per-expert weights are stacked into
//     grouped [N,...] constants and Gather(grouped_w, topk_indices, axis=0)
//     selects just those K on-device; the N-way masked Add chain becomes a
//     K-way score-weighted sum.
//   - multi-token chunk (batch > 1, or symbolically dynamic): different
//     tokens in the chunk route to different experts, so there is no single
//     K-of-N subset to Gather. All N experts are instead densely computed
//     for the whole chunk (batched MatMul with N as a leading batch axis)
//     and combined via a [chunk,N] routing-weight matrix built by
//     scattering each token's top-K weights into a zero background
//     (ScatterElementsUpdate).
//
// Disabled by default; enable via config key "enable_moe_gather" = "true"
// (the same flag gates both the single-token and multi-token-chunk rewrite
// strategies).
class MoEGatherRewrite : public ov::pass::ModelPass {
 public:
  OPENVINO_RTTI("MoEGatherRewrite");
  bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

// Scans |model| for Gemma4-style MoE router(s) and reports whether their
// token/chunk batch shape is a multi-token chunk (true) vs a single-token
// chunk (false, statically 1), using the same detection this pass relies
// on. Returns std::nullopt if no MoE router is found, or false if multiple
// routers disagree on the shape.
std::optional<bool> DetectMoeIsMultiTokenChunk(
    const std::shared_ptr<ov::Model>& model);

// Configurable runner for NPU-specific optimization passes.
// Use the setter APIs to toggle individual optimizations, then call Run() to
// apply the enabled passes to a model.
class NpuOptimizer {
 public:
  // Toggles the SplitSharedConstants pass. Enabled by default: giving every
  // consumer of a multi-consumer Constant its own copy is always safe and
  // guards every other pass (including NPUW's own later repeated-subgraph
  // matching) against pre-existing accidental sharing in the input graph.
  NpuOptimizer& SetSplitSharedConstants(bool enable) {
    split_shared_constants_ = enable;
    return *this;
  }

  // Toggles a ConstantFolding pass. Disabled by default.
  NpuOptimizer& SetConstantFold(bool enable) {
    constant_fold_ = enable;
    return *this;
  }

  // Toggles the EliminateMatMulFakeQuantize pass. Disabled by default.
  NpuOptimizer& SetEliminateMatMulFakeQuantize(bool enable) {
    eliminate_matmul_fq_ = enable;
    return *this;
  }

  // Toggles the CastIntegerSignToFloat pass. Enabled by default because the
  // NPU plugin cannot lower integer Sign operations.
  NpuOptimizer& SetCastIntegerSignToFloat(bool enable) {
    cast_integer_sign_to_float_ = enable;
    return *this;
  }

  // Toggles the FuseSplitAttentionToSDPA pass. Disabled by default; enable
  // via config key "fuse_split_attention_to_sdpa" = "true" to collapse the
  // Gemma-style split-attention pattern into a single SDPA op.
  NpuOptimizer& SetFuseSplitAttentionToSDPA(bool enable) {
    fuse_split_attention_to_sdpa_ = enable;
    return *this;
  }

  // Controls whether the SDPA fusion pads an unaligned merged KV sequence
  // length up to the NPU kernel's required alignment.
  NpuOptimizer& SetSdpaPadKvToAlignment(bool enable) {
    sdpa_pad_kv_to_alignment_ = enable;
    return *this;
  }

  // Toggles the MoEGatherRewrite pass. Disabled by default; enable via config
  // key "enable_moe_gather" = "true" to turn Gemma4's dense masked MoE into
  // gather-based selective (K-of-N) expert computation.
  NpuOptimizer& SetEnableMoeGather(bool enable) {
    enable_moe_gather_ = enable;
    return *this;
  }

  // Runs all currently-enabled passes on |model|.
  void Run(const std::shared_ptr<ov::Model>& model) const;

 private:
  bool split_shared_constants_ = true;
  bool constant_fold_ = false;
  bool eliminate_matmul_fq_ = false;
  bool cast_integer_sign_to_float_ = true;
  bool fuse_split_attention_to_sdpa_ = false;
  bool sdpa_pad_kv_to_alignment_ = true;
  bool enable_moe_gather_ = false;
};

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_NPU_OPTIMIZER_H_
