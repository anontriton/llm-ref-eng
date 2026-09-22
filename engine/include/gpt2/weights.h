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
  const float* data = nullptr;       // set only when the tensor is float32
  const int8_t* qdata = nullptr;     // set only when the tensor is int8
  const int32_t* idata = nullptr;    // set only when the tensor is int32
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
//
// Outlier columns (Phase 5, wte only). ln_f's output has a few hidden dims two
// hundred times the median, and an int8 error in those columns of the tied
// lm_head is multiplied by them; reference/wte_sim.py measured that as the
// whole of int8 wte's damage. So those columns are held back in fp32: the int8
// table stores zeros there -- which also keeps them out of the row scales --
// and `outlier` holds the real values, [rows, n_outlier] row-major, for the
// ascending column indices in `outlier_cols`. The stored value is then
//     w[j, c] ~= q[j, c] * scale[j]           for c not an outlier column
//     w[j, c]  = outlier[j, k]                 for c = outlier_cols[k]
struct Matrix {
  const float* f32 = nullptr;
  const int8_t* i8 = nullptr;
  const float* scale = nullptr;
  const int32_t* outlier_cols = nullptr;
  const float* outlier = nullptr;
  int n_outlier = 0;
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

  // The same parse over bytes already in memory, which the Weights takes
  // ownership of. load() is this after reading the file; the browser build
  // streams the download straight into `blob` instead, since it has no file
  // to read. `name` stands in for the path in path() and in error messages.
  static Weights from_blob(std::vector<unsigned char> blob, const std::string& name);

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
  // "fp32", "int8" (layers only, wte fp32 -- the Phase 4 file), "int8-wte"
  // (wte too), or "int8-wte-o<k>" (wte with k columns held back in fp32).
  // Derived from what the file holds, so a result cannot misname it.
  const std::string& policy() const { return policy_; }
  const float* wpe() const { return wpe_; }
  const float* ln_f_w() const { return ln_f_w_; }
  const float* ln_f_b() const { return ln_f_b_; }
  const LayerWeights& layer(int i) const { return layers_.at(static_cast<size_t>(i)); }

 private:
  void resolve();
  void resolve_wte_outliers();

  std::string path_;
  std::vector<unsigned char> blob_;
  std::unordered_map<std::string, WeightView> views_;
  Config config_;
  std::string source_sha256_;

  Matrix wte_;
  uint32_t quant_ = 0;
  std::string policy_;
  const float* wpe_ = nullptr;
  const float* ln_f_w_ = nullptr;
  const float* ln_f_b_ = nullptr;
  std::vector<LayerWeights> layers_;
};

}  // namespace gpt2
