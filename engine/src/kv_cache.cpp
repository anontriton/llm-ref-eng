#include "gpt2/kv_cache.h"

#include <stdexcept>
#include <string>

namespace gpt2 {

KVCache::KVCache(const Config& cfg, int capacity)
    : capacity_(capacity), d_model_(cfg.d_model) {
  if (capacity <= 0) {
    throw std::runtime_error("KVCache: capacity must be positive");
  }
  if (capacity > cfg.n_ctx) {
    throw std::runtime_error("KVCache: capacity " + std::to_string(capacity) +
                             " exceeds context " + std::to_string(cfg.n_ctx));
  }
  const size_t n = static_cast<size_t>(cfg.n_layer) * capacity * cfg.d_model;
  k_.assign(n, 0.0f);
  v_.assign(n, 0.0f);
}

int KVCache::extend(int n) {
  if (n <= 0) throw std::runtime_error("KVCache::extend: n must be positive");
  if (size_ + n > capacity_) {
    throw std::runtime_error(
        "KVCache::extend: " + std::to_string(size_) + " + " +
        std::to_string(n) + " exceeds capacity " + std::to_string(capacity_));
  }
  const int start = size_;
  size_ += n;
  return start;
}

}  // namespace gpt2
