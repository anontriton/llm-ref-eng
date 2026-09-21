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
  const float* data = nullptr;       // null when the tensor is int8
  const int8_t* qdata = nullptr;     // null when the tensor is float32
  std::vector<int64_t> shape;
  int64_t numel = 0;
  bool quantized() const { return qdata != nullptr; }
};

// A weight matrix the kernels multiply by, in whichever representation the
// file happened to carry. Phase 4 introduced the second one; everything above
// the kernels stops caring which it is.
//
// When quantized, the stored value is w ~= q * scale, with one scale per
// output channel -- per column for the [in, out] projections, per row for wte,
// whose output channel as the tied lm_head is the vocabulary. The scale is
// applied once after the reduction rather than per element, which is both
// cheaper and slightly more accurate than dequantizing first.
struct Matrix {
  const float* f32 = nullptr;
  const int8_t* i8 = nullptr;
  const float* scale = nullptr;
  bool quantized() const { return i8 != nullptr; }
};

// Everything one transformer block needs, resolved once at load time so the
// forward pass does no string lookups.
struct LayerWeights {
  const float* ln_1_w = nullptr;
  const float* ln_1_b = nullptr;
  Matrix c_attn_w;                   // [d_model, 3 * d_model], fused QKV
  const float* c_attn_b = nullptr;   // [3 * d_model]
  Matrix attn_proj_w;                // [d_model, d_model]
  const float* attn_proj_b = nullptr;
  const float* ln_2_w = nullptr;
  const float* ln_2_b = nullptr;
  Matrix c_fc_w;                     // [d_model, d_ff]
  const float* c_fc_b = nullptr;
  Matrix mlp_proj_w;                 // [d_ff, d_model]
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

  // wte does double duty: the tied lm_head matrix, and the embedding table.
  const Matrix& wte() const { return wte_; }

  // Write token `id`'s embedding into `dst` (d_model floats), dequantizing if
  // the table is int8.
  void embed(int32_t id, float* dst) const;

  // True when the file declared itself quantized; recorded by the eval
  // manifest so a result cannot misreport what it ran.
  bool quantized() const { return quant_ != 0; }
  const char* policy() const { return quant_ == 0 ? "fp32" : "int8"; }
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

  Matrix wte_;
  uint32_t quant_ = 0;
  const float* wpe_ = nullptr;
  const float* ln_f_w_ = nullptr;
  const float* ln_f_b_ = nullptr;
  std::vector<LayerWeights> layers_;
};

}  // namespace gpt2
