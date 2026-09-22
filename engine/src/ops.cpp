#include "gpt2/ops.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "gpt2/backend/backend.h"
#include "gpt2/threading.h"

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

// Grow the merge stack to hold `need` floats at this level. The buffers live
// in thread-local storage and are reused across calls, so a parallel linear()
// does not allocate per tile; sizes differ between call sites, hence the grow.
Partial& level(std::vector<Partial>& stack, size_t depth, size_t need) {
  if (depth == stack.size()) stack.emplace_back();
  Partial& p = stack[depth];
  if (p.acc.size() < need) p.acc.resize(need);
  return p;
}

// How many column chunks to cut the output into.
//
// One, whenever there are already enough row blocks to keep every thread busy:
// a column chunk reads W with a stride, which is the pattern that measured
// slower before blocking, so it is a last resort rather than a default. It
// exists for decode, which is a single row and therefore a single row block --
// without it, decode would have no parallelism at all.
int column_chunks(int rows, int block, int n_out, int workers) {
  if (workers <= 1) return 1;
  const int blocks = (rows + block - 1) / block;
  if (blocks >= workers) return 1;
  int chunks = (workers + blocks - 1) / blocks;
  chunks = std::min(chunks, n_out / 8);
  return std::max(chunks, 1);
}

int strip_width(int n_in, int n_out, int rows) {
  if (rows < kMinTiledRows) return n_out;
  const size_t per_column = static_cast<size_t>(n_in) * sizeof(float);
  size_t strip = kStripBytes / (per_column ? per_column : 1);
  strip = (strip / 8) * 8;
  if (strip < 8) strip = 8;
  return static_cast<int>(std::min<size_t>(strip, static_cast<size_t>(n_out)));
}

}  // namespace

void linear(const float* x, const Matrix& W, const float* bias, float* out,
            int rows, int n_in, int n_out) {
  // One row at a time reproduces the original loop exactly, which is what
  // decode gets.
  const int block = std::min(kRowBlock, rows);
  const int blocks = (rows + block - 1) / block;
  const int chunks = column_chunks(rows, block, n_out, threads::count());
  const int per_chunk = chunks > 1 ? ((n_out / chunks) / 8) * 8 : n_out;

  // One tile of the output: rows [r0, r0+rb) by columns [c0, c1). Tiles are
  // disjoint and share nothing, so running them on different threads cannot
  // change a single bit of the answer -- and at one chunk and one thread this
  // is the serial loop it replaced, unchanged.
  const auto tile = [&](int index) {
    const int b = index / chunks;
    const int c = index % chunks;
    const int r0 = b * block;
    const int rb = std::min(block, rows - r0);
    const int c0 = c * per_chunk;
    const int c1 = (c == chunks - 1) ? n_out : (c0 + per_chunk);
    const size_t width = static_cast<size_t>(c1 - c0);
    const size_t span = static_cast<size_t>(rb) * width;

    static thread_local std::vector<Partial> stack;
    size_t depth = 0;

    for (int i0 = 0; i0 < n_in; i0 += kLeaf) {
      const int i1 = std::min(i0 + kLeaf, n_in);

      Partial& leaf = level(stack, depth, static_cast<size_t>(block) * width);
      std::fill(leaf.acc.begin(), leaf.acc.begin() + span, 0.0f);
      // W's row for input i is read once and spent across every row in the
      // block, instead of being re-read for each row of x.
      for (int i = i0; i < i1; ++i) {
        const size_t off = static_cast<size_t>(i) * n_out + c0;
        for (int rr = 0; rr < rb; ++rr) {
          const float a = x[static_cast<size_t>(r0 + rr) * n_in + i];
          float* dst = leaf.acc.data() + static_cast<size_t>(rr) * width;
          if (W.quantized()) {
            backend::axpy_i8(a, W.i8 + off, dst, width);
          } else {
            backend::axpy(a, W.f32 + off, dst, width);
          }
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

    // The int8 scale lands here, once per output element, after the whole
    // reduction -- not folded into each product. One multiply instead of
    // n_in of them, and one rounding instead of n_in.
    const float* scale = W.quantized() ? W.scale + c0 : nullptr;
    for (int rr = 0; rr < rb; ++rr) {
      float* orow = out + static_cast<size_t>(r0 + rr) * n_out + c0;
      const float* acc = stack[0].acc.data() + static_cast<size_t>(rr) * width;
      if (scale != nullptr) {
        if (bias != nullptr) {
          for (size_t j = 0; j < width; ++j) orow[j] = acc[j] * scale[j] + bias[c0 + j];
        } else {
          for (size_t j = 0; j < width; ++j) orow[j] = acc[j] * scale[j];
        }
      } else if (bias != nullptr) {
        for (size_t j = 0; j < width; ++j) orow[j] = acc[j] + bias[c0 + j];
      } else {
        for (size_t j = 0; j < width; ++j) orow[j] = acc[j];
      }
    }
  };

  threads::parallel_for(blocks * chunks, tile);
}

void linear_tied(const float* x, const Matrix& Wt, float* out,
                 int rows, int n_in, int n_out) {
  // lm_head is the worst offender: Wt is the 50257 x 768 embedding table, 154
  // MB, and walking it once per row was 20 GB of traffic on its own. Same fix,
  // and trivially exact here -- each output element is still one dot product
  // over the same span; only the order the elements are visited in changes.
  const int strip = strip_width(n_in, n_out, rows);
  const int strips = (n_out + strip - 1) / strip;

  // Each strip owns a disjoint set of output columns, which is already the
  // unit of parallelism; lm_head cuts into 300 of them.
  threads::parallel_for(strips, [&](int s) {
    const int j0 = s * strip;
    const int j1 = std::min(j0 + strip, n_out);

    // Outlier columns, when Wt holds any: x's values at those columns,
    // gathered once per row so each output adds one short contiguous dot.
    static thread_local std::vector<float> xo;
    const size_t k = static_cast<size_t>(Wt.n_outlier);
    xo.resize(k);

    for (int r = 0; r < rows; ++r) {
      const float* xr = x + static_cast<size_t>(r) * n_in;
      float* orow = out + static_cast<size_t>(r) * n_out;
      if (Wt.quantized() && k > 0) {
        for (size_t i = 0; i < k; ++i) xo[i] = xr[Wt.outlier_cols[i]];
        // The int8 table is zero in the outlier columns (the loader checks),
        // so its dot product runs over all n_in unchanged and those terms add
        // nothing; the fp32 columns follow as their own reduction through
        // the same backend, which keeps every backend bit-identical.
        for (int j = j0; j < j1; ++j) {
          orow[j] = backend::dot_i8(xr, Wt.i8 + static_cast<size_t>(j) * n_in,
                                    static_cast<size_t>(n_in)) * Wt.scale[j] +
                    backend::dot(xo.data(), Wt.outlier + static_cast<size_t>(j) * k, k);
        }
      } else if (Wt.quantized()) {
        for (int j = j0; j < j1; ++j) {
          orow[j] = backend::dot_i8(xr, Wt.i8 + static_cast<size_t>(j) * n_in,
                                    static_cast<size_t>(n_in)) * Wt.scale[j];
        }
      } else {
        for (int j = j0; j < j1; ++j) {
          orow[j] = backend::dot(xr, Wt.f32 + static_cast<size_t>(j) * n_in,
                                 static_cast<size_t>(n_in));
        }
      }
    }
  });
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
