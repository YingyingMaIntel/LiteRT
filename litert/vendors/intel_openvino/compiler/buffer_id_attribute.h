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

#ifndef LITERT_VENDORS_INTEL_OPENVINO_COMPILER_BUFFER_ID_ATTRIBUTE_H_
#define LITERT_VENDORS_INTEL_OPENVINO_COMPILER_BUFFER_ID_ATTRIBUTE_H_

#include <cstdint>

#include "openvino/core/runtime_attribute.hpp"
#include "openvino/core/rtti.hpp"

namespace litert {
namespace openvino {

// Marks an ov::op::v0::Constant as backed by WeightBank buffer |buffer_id|
// (see weight_bank.h). HarvestSharedConstants (alias_shared_constants.h)
// stamps this once, right after NPU optimization, so later passes resolve
// pool identity by reading the graph itself instead of guessing it back from
// a name that may not have survived intervening transforms.
//
// Not copyable: a node produced by replacing/cloning a tagged Constant must
// NOT silently inherit its source's identity (that would be a stale/wrong
// buffer_id) -- whatever pass creates such a node must explicitly re-tag it
// via HarvestSharedConstants. No attribute present is the safe default
// ("not resolved to a shared buffer").
class LiteRtBufferIdAttribute : public ov::RuntimeAttribute {
 public:
  OPENVINO_RTTI("LiteRtBufferIdAttribute", "0", ov::RuntimeAttribute);

  LiteRtBufferIdAttribute() = delete;
  explicit LiteRtBufferIdAttribute(int32_t buffer_id) : buffer_id(buffer_id) {}

  bool is_copyable() const override { return false; }

  int32_t buffer_id;  // NOLINT(*-readability-class-member-naming)
};

}  // namespace openvino
}  // namespace litert

#endif  // LITERT_VENDORS_INTEL_OPENVINO_COMPILER_BUFFER_ID_ATTRIBUTE_H_
