// Attention keys and values, retained across decode steps.
//
// Without this, generating token t+1 re-runs tokens 0..t through all 12 layers,
// so per-token cost grows with sequence length: the Phase 2 baseline spends
// 8813 ms/token at T=128..255, more than the 5810 ms it takes to process the
// entire prompt. Almost all of that is recomputation.
//
// What makes the cache correct is causality. Position t attends only to
// positions <= t, so the k and v a position contributes never change once
// computed -- appending a token cannot alter what came before it. They are
// therefore computed once and kept.
//
// Layout is [layer][pos][d_model], so a single head's slice at one position is
// contiguous and the dot products in attention stay contiguous too, exactly as
// they were when k and v were read out of the fused qkv buffer.
#pragma once

#include <cstddef>
#include <vector>

#include "gpt2/config.h"

namespace gpt2 {

class KVCache {
 public:
  // `capacity` is the number of positions this cache can hold. Sizing it to
  // the prompt is right for a one-shot forward; sizing it to cfg.n_ctx is
  // right for generation, and costs 2 * n_layer * n_ctx * d_model floats --
  // 72 MiB for GPT-2 124M at the full 1024-token context.
  KVCache(const Config& cfg, int capacity);

  // A cache sized to the full context window.
  explicit KVCache(const Config& cfg) : KVCache(cfg, cfg.n_ctx) {}

  int capacity() const { return capacity_; }

  // Positions filled so far; also the absolute position the next token takes.
  int size() const { return size_; }

  void clear() { size_ = 0; }

  // Make room for `n` more positions and return the absolute index of the
  // first. Throws std::runtime_error if that would exceed capacity.
  int extend(int n);

  float* k(int layer, int pos) { return &k_[offset(layer, pos)]; }
  float* v(int layer, int pos) { return &v_[offset(layer, pos)]; }
  const float* k(int layer, int pos) const { return &k_[offset(layer, pos)]; }
  const float* v(int layer, int pos) const { return &v_[offset(layer, pos)]; }

 private:
  size_t offset(int layer, int pos) const {
    return (static_cast<size_t>(layer) * capacity_ + pos) * d_model_;
  }

  int capacity_;
  int d_model_;
  int size_ = 0;
  std::vector<float> k_;
  std::vector<float> v_;
};

}  // namespace gpt2
