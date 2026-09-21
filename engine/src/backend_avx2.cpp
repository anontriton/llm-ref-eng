// AVX2 backend.
//
// Reproduces backend_scalar.cpp bit for bit. That is the whole design of this
// layer: kAccumLanes is 8 because that is one AVX2 register of floats, so the
// scalar backend's eight accumulator lanes ARE this register, and the same
// terms land in the same lane in the same order.
//
// Deliberately no FMA, which is the one thing that would break it. The scalar
// spelling `lane[k] += a[i] * b[i]` rounds the product to fp32 and then rounds
// the sum; _mm256_fmadd_ps rounds once. The fused form is faster and slightly
// more accurate, and it would make this backend disagree with the scalar one
// and with the wasm_simd128 backend in Phase 5, which has no FMA to offer.
// Agreement is worth more here than the instruction: it is what lets the
// oracle validate one backend and mean it about the others. The build compiles
// this file without -mfma and with -ffp-contract=off so the compiler cannot
// reintroduce it behind the intrinsics.
//
// Only this file is built with -mavx2. The kernels in ops.cpp stay baseline
// and reach the vector units through these functions, which is what "SIMD goes
// behind an abstraction layer" means in practice.
#include "gpt2/backend/backend.h"

#include <immintrin.h>

#include <cstdio>
#include <cstdlib>
#include <limits>

static_assert(gpt2::backend::kAccumLanes == 8,
              "the AVX2 backend is written against an 8-lane accumulator, "
              "which is exactly one __m256");

namespace gpt2::backend {

const char* name() { return "avx2"; }

namespace {

// A binary built for AVX2 on a machine without it dies on the first vector
// instruction with SIGILL and no explanation. Say so instead.
struct CpuCheck {
  CpuCheck() {
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2")) {
      std::fprintf(stderr,
                   "this binary was built with the AVX2 backend, but this CPU "
                   "does not support AVX2; rebuild with -DGPT2_BACKEND=scalar\n");
      std::abort();
    }
  }
};
const CpuCheck cpu_check;

// The scalar reduce_lanes, instruction for instruction:
//   half=4:  lane[i] += lane[i + 4]   for i in 0..3
//   half=2:  lane[i] += lane[i + 2]   for i in 0..1
//   half=1:  lane[0] += lane[1]
// Each step is an add of the same two values the scalar loop adds, so the
// rounding at every node of the tree is the same.
inline float reduce_lanes(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);         // lane[0..3]
  const __m128 hi = _mm256_extractf128_ps(v, 1);       // lane[4..7]
  const __m128 s4 = _mm_add_ps(lo, hi);                // half = 4
  const __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));  // half = 2
  const __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(s1);                            // half = 1
}

// The tail is at most seven elements and joins the lanes it lines up with,
// exactly as the scalar backend does. Doing it scalar-side rather than with a
// masked load keeps that correspondence obvious, and costs nothing: the shapes
// this engine runs are multiples of eight, so the tail is almost always empty.
inline float finish_tail(__m256 acc, const float* a, const float* b,
                         size_t body, size_t tail) {
  if (tail == 0) return reduce_lanes(acc);
  alignas(32) float lane[kAccumLanes];
  _mm256_store_ps(lane, acc);
  for (size_t i = 0; i < tail; ++i) {
    lane[i] += (b == nullptr) ? a[body + i] : a[body + i] * b[body + i];
  }
  return reduce_lanes(_mm256_load_ps(lane));
}

}  // namespace

float dot(const float* a, const float* b, size_t n) {
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  __m256 acc = _mm256_setzero_ps();
  for (size_t i = 0; i < body; i += kAccumLanes) {
    // Multiply, then add. Two roundings, as above.
    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                           _mm256_loadu_ps(b + i)));
  }
  return finish_tail(acc, a, b, body, tail);
}

void axpy(float alpha, const float* x, float* y, size_t n) {
  // Elementwise, so there is no summation order to preserve -- only the
  // requirement that each y[i] is still a rounded product added to a rounded
  // accumulator.
  const __m256 va = _mm256_set1_ps(alpha);
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    _mm256_storeu_ps(y + i,
                     _mm256_add_ps(_mm256_loadu_ps(y + i),
                                   _mm256_mul_ps(va, _mm256_loadu_ps(x + i))));
  }
  for (size_t i = body; i < n; ++i) y[i] += alpha * x[i];
}

float sum(const float* x, size_t n) {
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  __m256 acc = _mm256_setzero_ps();
  for (size_t i = 0; i < body; i += kAccumLanes) {
    acc = _mm256_add_ps(acc, _mm256_loadu_ps(x + i));
  }
  return finish_tail(acc, x, nullptr, body, tail);
}

float max(const float* x, size_t n) {
  // Left scalar on purpose. _mm256_max_ps does not agree with `>` on NaN or on
  // signed zero, and this is not a hot loop -- softmax calls it once per row.
  // A vector spelling would be trading a real correspondence for nothing.
  float m = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < n; ++i) {
    if (x[i] > m) m = x[i];
  }
  return m;
}

void scale(float* x, float alpha, size_t n) {
  const __m256 va = _mm256_set1_ps(alpha);
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    _mm256_storeu_ps(x + i, _mm256_mul_ps(_mm256_loadu_ps(x + i), va));
  }
  for (size_t i = body; i < n; ++i) x[i] *= alpha;
}

void add(const float* x, float* y, size_t n) {
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    _mm256_storeu_ps(y + i, _mm256_add_ps(_mm256_loadu_ps(y + i),
                                          _mm256_loadu_ps(x + i)));
  }
  for (size_t i = body; i < n; ++i) y[i] += x[i];
}

}  // namespace gpt2::backend
