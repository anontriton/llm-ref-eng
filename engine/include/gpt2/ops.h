// Kernels. Each one mirrors a specific piece of reference/model.py.
//
// Every buffer is contiguous row-major fp32. Kernels call into
// gpt2::backend for their inner loops and contain no intrinsics themselves,
// so Phase 3's AVX2 backend and Phase 5's wasm_simd128 backend drop in
// underneath without touching this file.
#pragma once

#include <cstddef>

#include "gpt2/config.h"

namespace gpt2::ops {

// LayerNorm over the last dimension, with the biased (population) variance --
// divide by N, not N-1, matching torch.var(unbiased=False).
//   out[r, :] = (x[r, :] - mean) / sqrt(var + eps) * w + b
void layernorm(const float* x, const float* w, const float* b, float* out,
               int rows, int n, float eps);

// y = x @ W + bias, with W stored [n_in, n_out] -- the checkpoint's own
// orientation (HF calls it Conv1D). No transpose happens anywhere in this
// engine; that is the point of storing it this way.
//
// Accumulates by sweeping W's rows, which keeps every memory access
// contiguous: out[r, :] starts at bias and takes one axpy per input element.
void linear(const float* x, const float* W, const float* bias, float* out,
            int rows, int n_in, int n_out);

// y = x @ Wt^T + 0, with Wt stored [n_out, n_in]: the tied lm_head, where the
// weight is wte and the natural inner loop is a contiguous dot product.
void linear_tied(const float* x, const float* Wt, float* out,
                 int rows, int n_in, int n_out);

// The tanh approximation of GELU, which is what GPT-2 was trained with. NOT the
// erf formulation -- the two differ by ~1e-3 in the tails, an order of
// magnitude above the 1e-4 tolerance.
void gelu_new(const float* x, float* out, size_t n);

// Row-wise softmax in place, shifted by the row max for stability. A row entry
// of -inf produces exactly 0, which is how masked positions vanish.
void softmax_rows(float* x, int rows, int n);

}  // namespace gpt2::ops
