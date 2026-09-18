#include "gpt2/ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "gpt2/backend/backend.h"

namespace gpt2::ops {

void layernorm(const float* x, const float* w, const float* b, float* out,
               int rows, int n, float eps) {
  const float inv_n = 1.0f / static_cast<float>(n);
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<size_t>(r) * n;
    float* orow = out + static_cast<size_t>(r) * n;

    const float mean = backend::sum(xr, static_cast<size_t>(n)) * inv_n;

    // Variance in the same two-pass form the reference uses: torch computes
    // the mean, then the mean of the squared deviations. A one-pass
    // sum-of-squares formulation is algebraically equal and numerically worse,
    // and the difference is visible at this tolerance.
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float d = xr[i] - mean;
      acc += d * d;
    }
    const float inv_std = 1.0f / std::sqrt(acc * inv_n + eps);

    for (int i = 0; i < n; ++i) orow[i] = (xr[i] - mean) * inv_std * w[i] + b[i];
  }
}

namespace {

// Pairwise (binary-tree) accumulation along n_in.
//
// linear() sums up to 3072 products per output element. One sequential fp32
// chain loses roughly n*eps, which measured ~10x worse than the reference's
// blocked GEMM and pushed the residual stream well past tolerance. Pairwise
// summation brings that to O(log n * eps) for the cost of a handful of
// buffers, and -- unlike accumulating in double -- it is something the AVX2 and
// wasm backends can reproduce exactly, so Phase 3 will not regress.
//
// The shape of the tree is fixed by kLeaf and the merge rule below, never by
// the data, so the result is deterministic. This lives in the kernel rather
// than the backend because it is a choice about summation order; the backend's
// job is only to sweep the span it is handed.
constexpr int kLeaf = 64;

// One level of the pending merge stack: a partial sum and how many leaves it
// covers. Two entries merge only when they cover the same count, which is what
// makes the tree balanced and the order reproducible.
struct Partial {
  std::vector<float> acc;
  int leaves = 0;
};

}  // namespace

void linear(const float* x, const float* W, const float* bias, float* out,
            int rows, int n_in, int n_out) {
  const size_t width = static_cast<size_t>(n_out);
  std::vector<Partial> stack;

  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<size_t>(r) * n_in;
    float* orow = out + static_cast<size_t>(r) * n_out;

    size_t depth = 0;  // live entries in `stack`; buffers are reused across rows

    for (int i0 = 0; i0 < n_in; i0 += kLeaf) {
      const int i1 = std::min(i0 + kLeaf, n_in);

      if (depth == stack.size()) stack.push_back(Partial{std::vector<float>(width), 0});
      Partial& leaf = stack[depth];
      std::fill(leaf.acc.begin(), leaf.acc.end(), 0.0f);
      for (int i = i0; i < i1; ++i) {
        backend::axpy(xr[i], W + static_cast<size_t>(i) * n_out, leaf.acc.data(), width);
      }
      leaf.leaves = 1;
      ++depth;

      // Merge equal-weight neighbours, which keeps the stack O(log n) deep.
      while (depth >= 2 && stack[depth - 1].leaves == stack[depth - 2].leaves) {
        backend::add(stack[depth - 1].acc.data(), stack[depth - 2].acc.data(), width);
        stack[depth - 2].leaves *= 2;
        --depth;
      }
    }

    // Fold whatever is left, smallest first: the tail blocks are the lightest,
    // so adding them before the heavy ones loses the least.
    for (size_t k = depth; k-- > 1;) {
      backend::add(stack[k].acc.data(), stack[k - 1].acc.data(), width);
    }

    if (bias != nullptr) {
      for (int j = 0; j < n_out; ++j) orow[j] = stack[0].acc[static_cast<size_t>(j)] + bias[j];
    } else {
      for (int j = 0; j < n_out; ++j) orow[j] = stack[0].acc[static_cast<size_t>(j)];
    }
  }
}

void linear_tied(const float* x, const float* Wt, float* out,
                 int rows, int n_in, int n_out) {
  for (int r = 0; r < rows; ++r) {
    const float* xr = x + static_cast<size_t>(r) * n_in;
    float* orow = out + static_cast<size_t>(r) * n_out;
    for (int j = 0; j < n_out; ++j) {
      orow[j] = backend::dot(xr, Wt + static_cast<size_t>(j) * n_in,
                             static_cast<size_t>(n_in));
    }
  }
}

void gelu_new(const float* x, float* out, size_t n) {
  // sqrt(2/pi), the constant in the tanh approximation.
  const float c = 0.7978845608028654f;
  for (size_t i = 0; i < n; ++i) {
    const float v = x[i];
    const float inner = c * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + std::tanh(inner));
  }
}

void softmax_rows(float* x, int rows, int n) {
  for (int r = 0; r < rows; ++r) {
    float* row = x + static_cast<size_t>(r) * n;
    const float m = backend::max(row, static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) row[i] = std::exp(row[i] - m);
    const float s = backend::sum(row, static_cast<size_t>(n));
    backend::scale(row, 1.0f / s, static_cast<size_t>(n));
  }
}

}  // namespace gpt2::ops
