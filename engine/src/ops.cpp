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

// Cache blocking for the two matmuls, in the two different shapes their weight
// layouts demand. Both used to walk their whole weight matrix once per row, so
// a 128-token prefill dragged every W through cache 128 times.
//
// The layouts are not the same, and neither is the fix. Measured at T=128,
// summed over a full prefill's worth of calls:
//
//                        per row   strips   row blocks
//     qkv       768->2304   973 ms   1015 ms    946 ms
//     attn_proj  768->768   310 ms    333 ms    317 ms
//     c_fc      768->3072  1585 ms   1553 ms   1292 ms
//     mlp_proj  3072->768  1586 ms   1892 ms   1295 ms
//     lm_head  768->50257  1214 ms    582 ms      --
//
// linear()'s W is row-major along n_in, so a strip of output columns reads it
// with a stride of n_out floats and four of the five shapes got *slower*.
// Blocking rows instead keeps each W row contiguous and shares it across the
// block, which is where the gain is.
//
// linear_tied()'s Wt is the transpose -- each output column is one contiguous
// row of it -- so there a column strip is exactly the contiguous case, and
// lm_head halves.
//
// Neither touches the reduction along n_in: leaf size and merge tree are
// unchanged, every output element is still the same terms in the same order,
// and the result is bit-identical. Blocking n_in would reshape the tree and
// change the answer.

// Rows carried together in linear(). Eight keeps the accumulators near 0.5 MiB
// at the widest n_out; 32 measured ~1.6% faster for four times the footprint,
// which stops fitting L2 on a smaller machine.
constexpr int kRowBlock = 8;

// Tiling only pays when several rows reuse the tile, and decode is always one
// row.
constexpr int kMinTiledRows = 4;

// Width of the output strip in linear_tied(), sized so n_in x strip floats sit
// inside a typical L2 alongside x and the output.
constexpr size_t kStripBytes = 512 * 1024;

int strip_width(int n_in, int n_out, int rows) {
  if (rows < kMinTiledRows) return n_out;
  const size_t per_column = static_cast<size_t>(n_in) * sizeof(float);
  size_t strip = kStripBytes / (per_column ? per_column : 1);
  strip = (strip / 8) * 8;
  if (strip < 8) strip = 8;
  return static_cast<int>(std::min<size_t>(strip, static_cast<size_t>(n_out)));
}

}  // namespace

void linear(const float* x, const float* W, const float* bias, float* out,
            int rows, int n_in, int n_out) {
  std::vector<Partial> stack;
  // One row at a time reproduces the original loop exactly, which is what
  // decode gets.
  const int block = std::min(kRowBlock, rows);

  for (int r0 = 0; r0 < rows; r0 += block) {
    const int rb = std::min(block, rows - r0);
    const size_t span = static_cast<size_t>(rb) * n_out;

    size_t depth = 0;  // live entries in `stack`; buffers are reused

    for (int i0 = 0; i0 < n_in; i0 += kLeaf) {
      const int i1 = std::min(i0 + kLeaf, n_in);

      if (depth == stack.size()) {
        stack.push_back(
            Partial{std::vector<float>(static_cast<size_t>(block) * n_out), 0});
      }
      Partial& leaf = stack[depth];
      std::fill(leaf.acc.begin(), leaf.acc.begin() + span, 0.0f);
      // W's row for input i is read once and spent across every row in the
      // block, instead of being re-read for each row of x.
      for (int i = i0; i < i1; ++i) {
        const float* wrow = W + static_cast<size_t>(i) * n_out;
        for (int rr = 0; rr < rb; ++rr) {
          backend::axpy(x[static_cast<size_t>(r0 + rr) * n_in + i], wrow,
                        leaf.acc.data() + static_cast<size_t>(rr) * n_out,
                        static_cast<size_t>(n_out));
        }
      }
      leaf.leaves = 1;
      ++depth;

      // Merge equal-weight neighbours, which keeps the stack O(log n) deep.
      // The rows in the block are independent lanes of the same tree, so each
      // output element's chain is the one it had before blocking.
      while (depth >= 2 && stack[depth - 1].leaves == stack[depth - 2].leaves) {
        backend::add(stack[depth - 1].acc.data(), stack[depth - 2].acc.data(), span);
        stack[depth - 2].leaves *= 2;
        --depth;
      }
    }

    // Fold whatever is left, smallest first: the tail blocks are the lightest,
    // so adding them before the heavy ones loses the least.
    for (size_t k = depth; k-- > 1;) {
      backend::add(stack[k].acc.data(), stack[k - 1].acc.data(), span);
    }

    for (int rr = 0; rr < rb; ++rr) {
      float* orow = out + static_cast<size_t>(r0 + rr) * n_out;
      const float* acc = stack[0].acc.data() + static_cast<size_t>(rr) * n_out;
      if (bias != nullptr) {
        for (int j = 0; j < n_out; ++j) orow[j] = acc[j] + bias[j];
      } else {
        for (int j = 0; j < n_out; ++j) orow[j] = acc[j];
      }
    }
  }
}

void linear_tied(const float* x, const float* Wt, float* out,
                 int rows, int n_in, int n_out) {
  // lm_head is the worst offender: Wt is the 50257 x 768 embedding table, 154
  // MB, and walking it once per row was 20 GB of traffic on its own. Same fix,
  // and trivially exact here -- each output element is still one dot product
  // over the same span; only the order the elements are visited in changes.
  const int strip = strip_width(n_in, n_out, rows);

  for (int j0 = 0; j0 < n_out; j0 += strip) {
    const int j1 = std::min(j0 + strip, n_out);

    for (int r = 0; r < rows; ++r) {
      const float* xr = x + static_cast<size_t>(r) * n_in;
      float* orow = out + static_cast<size_t>(r) * n_out;
      for (int j = j0; j < j1; ++j) {
        orow[j] = backend::dot(xr, Wt + static_cast<size_t>(j) * n_in,
                               static_cast<size_t>(n_in));
      }
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
