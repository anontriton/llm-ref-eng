// GPT-2 model geometry.
//
// Mirrors reference/model.py's GPT2Config field for field. The values are not
// defaulted here: they are read from the weight file, so a checkpoint and an
// engine can never silently disagree about the shape of the model.
#pragma once

#include <cstdint>

namespace gpt2 {

struct Config {
  int n_layer = 0;
  int n_head = 0;
  int d_model = 0;
  int d_ff = 0;
  int n_ctx = 0;
  int vocab_size = 0;

  // Held as f64 because the dump manifest reports it and oracle/compare.py
  // compares config values exactly; kernels take it as f32, which is what
  // PyTorch does when it adds a Python float to an fp32 tensor.
  double layer_norm_eps = 0.0;

  // Attention scales by 1/sqrt(d_head) -- the head dimension, not d_model.
  int d_head() const { return d_model / n_head; }
};

}  // namespace gpt2
