// Load the flat weight file written by scripts/convert_weights.py.
//
// The format is documented there and read here; the two must be changed
// together. Nothing else in the engine knows the on-disk layout.
//
// The file carries the sha256 of the safetensors checkpoint it came from. The
// engine copies that into its dump manifest without ever seeing the original
// checkpoint, which is what lets oracle/compare.py prove the engine and the
// reference ran on the same weights.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gpt2/config.h"

namespace gpt2 {

// A non-owning view of one tensor inside the loaded blob.
struct WeightView {
  const float* data = nullptr;
  std::vector<int64_t> shape;
  int64_t numel = 0;
};

// Everything one transformer block needs, resolved once at load time so the
// forward pass does no string lookups.
struct LayerWeights {
  const float* ln_1_w = nullptr;
  const float* ln_1_b = nullptr;
  const float* c_attn_w = nullptr;   // [d_model, 3 * d_model], fused QKV
  const float* c_attn_b = nullptr;   // [3 * d_model]
  const float* attn_proj_w = nullptr;  // [d_model, d_model]
  const float* attn_proj_b = nullptr;
  const float* ln_2_w = nullptr;
  const float* ln_2_b = nullptr;
  const float* c_fc_w = nullptr;     // [d_model, d_ff]
  const float* c_fc_b = nullptr;
  const float* mlp_proj_w = nullptr;  // [d_ff, d_model]
  const float* mlp_proj_b = nullptr;
};

class Weights {
 public:
  // Throws std::runtime_error on a bad magic, version, checksum field, or a
  // missing or misshapen tensor. Strictness here is the cheapest place to catch
  // the whole class of "the logits are slightly wrong" bugs.
  static Weights load(const std::string& path);

  const Config& config() const { return config_; }
  const std::string& source_sha256() const { return source_sha256_; }
  const std::string& path() const { return path_; }

  const WeightView& get(const std::string& name) const;
  bool has(const std::string& name) const { return views_.count(name) != 0; }
  size_t size() const { return views_.size(); }

  const float* wte() const { return wte_; }
  const float* wpe() const { return wpe_; }
  const float* ln_f_w() const { return ln_f_w_; }
  const float* ln_f_b() const { return ln_f_b_; }
  const LayerWeights& layer(int i) const { return layers_.at(static_cast<size_t>(i)); }

 private:
  void resolve();

  std::string path_;
  std::vector<unsigned char> blob_;
  std::unordered_map<std::string, WeightView> views_;
  Config config_;
  std::string source_sha256_;

  const float* wte_ = nullptr;
  const float* wpe_ = nullptr;
  const float* ln_f_w_ = nullptr;
  const float* ln_f_b_ = nullptr;
  std::vector<LayerWeights> layers_;
};

}  // namespace gpt2
