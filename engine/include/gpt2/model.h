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
#include "gpt2/kv_cache.h"
#include "gpt2/weights.h"

namespace gpt2 {

// Receives every intermediate activation, in forward order, under the oracle's
// naming scheme. `shape` is the reference's shape, batch dimension included.
using Tap = std::function<void(const std::string& name, const float* data,
                               const std::vector<int64_t>& shape)>;

// Which rows of logits a forward pass computes.
//
// All is the contract the oracle was written against: one row per input
// position, [T, vocab_size]. Last computes only the final row, [1, vocab_size]
// -- the only one generation reads. lm_head is a 768 x 50257 matmul per row,
// 31% of a 128-token prefill's arithmetic, and All spends it on rows nothing
// looks at. Every other tensor is computed exactly as under All, and so is the
// last row of logits: each logit is one dot product over the same span either
// way, so it is the same bits. The tap reports it as [1, 1, vocab_size].
enum class Logits { All, Last };

class Model {
 public:
  explicit Model(const Weights& weights);

  const Config& config() const { return cfg_; }

  // input_ids: T token ids. Returns logits, [T, vocab_size], row-major.
  // Throws std::runtime_error if T is 0 or exceeds the context window, or if a
  // token id is out of range -- an out-of-range id would otherwise read past
  // the embedding table and produce plausible-looking garbage.
  //
  // Stateless: every call starts from position 0. This is the form the oracle
  // validated in Phase 2 and it stays the definition of a correct forward
  // pass.
  std::vector<float> forward(const std::vector<int32_t>& input_ids,
                             const Tap& tap = nullptr,
                             Logits logits = Logits::All) const;

  // Incremental: appends `new_ids` after whatever `cache` already holds, and
  // returns logits for the new positions only -- [new_ids.size(), vocab_size].
  // Keys and values for earlier positions are read from the cache rather than
  // recomputed, which is the whole point.
  //
  // This is not a second implementation. It runs the same code as the call
  // above, which is just the special case of an empty cache; two forward
  // passes that must agree forever is exactly the arrangement that rots, and
  // the oracle only ever validates one of them.
  std::vector<float> forward(const std::vector<int32_t>& new_ids,
                             KVCache& cache, const Tap& tap = nullptr,
                             Logits logits = Logits::All) const;

 private:
  // Scratch buffers for one forward pass, sized once and shared by every
  // block, rather than allocated afresh in each of the twelve.
  struct Workspace;

  // `start_pos` is the absolute position of the first row of `new_ids`; the
  // cache supplies every position below it.
  std::vector<float> forward_impl(const std::vector<int32_t>& new_ids,
                                  int start_pos, KVCache& cache,
                                  const Tap& tap, Logits logits) const;

  void block(int i, float* x, int start_pos, int T_new, KVCache& cache,
             Workspace& ws, const Tap& tap) const;

  const Weights& w_;
  Config cfg_;
};

}  // namespace gpt2
