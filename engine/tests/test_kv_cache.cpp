// KVCache bookkeeping. The arithmetic the cache feeds is checked by
// oracle/compare.py against the reference; what is checked here is the
// indexing and the bounds, where an off-by-one would not crash but would
// silently attend to the wrong position.
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "gpt2/kv_cache.h"

namespace {

gpt2::Config small_config() {
  gpt2::Config cfg;
  cfg.n_layer = 3;
  cfg.n_head = 2;
  cfg.d_model = 8;
  cfg.d_ff = 32;
  cfg.n_ctx = 16;
  cfg.vocab_size = 100;
  cfg.layer_norm_eps = 1e-5;
  return cfg;
}

void test_extend() {
  const gpt2::Config cfg = small_config();
  gpt2::KVCache cache(cfg, 10);

  check::equal(cache.capacity(), 10, "capacity");
  check::equal(cache.size(), 0, "starts empty");

  check::equal(cache.extend(4), 0, "first extend starts at position 0");
  check::equal(cache.size(), 4, "size after extending by 4");
  check::equal(cache.extend(1), 4, "next extend starts where the last ended");
  check::equal(cache.size(), 5, "size after extending by 1");

  cache.clear();
  check::equal(cache.size(), 0, "clear rewinds to empty");
  check::equal(cache.extend(2), 0, "extend after clear starts at 0 again");
}

void test_capacity_is_enforced() {
  const gpt2::Config cfg = small_config();
  gpt2::KVCache cache(cfg, 4);
  cache.extend(4);

  bool threw = false;
  try {
    cache.extend(1);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check::ok(threw, "extending past capacity throws");
  check::equal(cache.size(), 4, "a rejected extend does not move size");

  threw = false;
  try {
    gpt2::KVCache too_big(cfg, cfg.n_ctx + 1);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check::ok(threw, "capacity beyond n_ctx throws");
}

// Layer and position must address disjoint storage. If they collided, layer 2
// would read layer 1's keys and the model would still run.
void test_slots_are_distinct() {
  const gpt2::Config cfg = small_config();
  const int cap = 6;
  gpt2::KVCache cache(cfg, cap);
  cache.extend(cap);

  for (int l = 0; l < cfg.n_layer; ++l) {
    for (int t = 0; t < cap; ++t) {
      const float tag = static_cast<float>(l * 100 + t);
      for (int c = 0; c < cfg.d_model; ++c) {
        cache.k(l, t)[c] = tag;
        cache.v(l, t)[c] = -tag;
      }
    }
  }

  for (int l = 0; l < cfg.n_layer; ++l) {
    for (int t = 0; t < cap; ++t) {
      const float tag = static_cast<float>(l * 100 + t);
      const std::string where =
          "layer " + std::to_string(l) + " pos " + std::to_string(t);
      check::close(cache.k(l, t)[0], tag, 0.0, "k intact at " + where);
      check::close(cache.k(l, t)[cfg.d_model - 1], tag,
                   0.0, "k intact at end of " + where);
      check::close(cache.v(l, t)[0], -tag, 0.0, "v intact at " + where);
    }
  }

  // k and v are separate stores, not two halves of one.
  check::ok(cache.k(0, 0) != cache.v(0, 0), "k and v do not alias");
}

}  // namespace

int main() {
  test_extend();
  test_capacity_is_enforced();
  test_slots_are_distinct();
  return check::report("test_kv_cache");
}
