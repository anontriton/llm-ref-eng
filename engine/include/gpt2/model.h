// The GPT-2 forward pass.
//
// A line-by-line port of reference/model.py. Where the two could reasonably
// differ -- variance formulation, GELU spelling, attention scale, the order the
// residual stream is accumulated in -- this file follows the reference, because
// the reference is the definition of correct.
//
// Batch is fixed at 1. The oracle's tensors carry a leading batch dimension, so
// the tap reports shapes as [1, T, ...] to match; the arithmetic never needs it.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "gpt2/config.h"
#include "gpt2/weights.h"

namespace gpt2 {

// Receives every intermediate activation, in forward order, under the oracle's
// naming scheme. `shape` is the reference's shape, batch dimension included.
using Tap = std::function<void(const std::string& name, const float* data,
                               const std::vector<int64_t>& shape)>;

class Model {
 public:
  explicit Model(const Weights& weights);

  const Config& config() const { return cfg_; }

  // input_ids: T token ids. Returns logits, [T, vocab_size], row-major.
  // Throws std::runtime_error if T is 0 or exceeds the context window, or if a
  // token id is out of range -- an out-of-range id would otherwise read past
  // the embedding table and produce plausible-looking garbage.
  std::vector<float> forward(const std::vector<int32_t>& input_ids,
                             const Tap& tap = nullptr) const;

 private:
  void block(int i, float* x, int T, const Tap& tap) const;

  const Weights& w_;
  Config cfg_;
};

}  // namespace gpt2
