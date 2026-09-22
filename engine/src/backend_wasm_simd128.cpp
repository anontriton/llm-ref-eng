// wasm_simd128 backend.
//
// Reproduces backend_scalar.cpp bit for bit, as the AVX2 backend does. The
// difference is width: a v128 holds four floats, so the scalar backend's eight
// accumulator lanes are two registers here -- `lo` is lane[0..3], `hi` is
// lane[4..7] -- and every step of the body is spent twice. Same terms, same
// lane, same order, same reduction tree. kAccumLanes was fixed at 8 in Phase 2
// so that this file could exist without changing a number.
//
// No FMA, and here that is not even a choice: wasm_simd128 has none (only the
// relaxed-simd proposal does, and its rounding is implementation-defined). The
// build still passes -ffp-contract=off, so the reason this file matches the
// scalar one is written down rather than inherited from the instruction set.
//
// Only this file is built with -msimd128. The kernels in ops.cpp stay baseline
// and reach the vector unit through these functions.
//
// Scalar transcendentals are not here and never were: tanh and exp live in
// ops.cpp and come from the platform libm. Under Emscripten that is musl, not
// glibc, so the bar this backend meets is the *wasm* scalar build, not the
// native one. web/README.md has the measurement.
#include "gpt2/backend/backend.h"

#include <wasm_simd128.h>

#include <limits>

static_assert(gpt2::backend::kAccumLanes == 8,
              "the wasm_simd128 backend is written against an 8-lane "
              "accumulator, which is exactly two v128 of f32");

namespace gpt2::backend {

const char* name() { return "wasm_simd128"; }

namespace {

// Eight accumulator lanes as two registers.
struct Acc {
  v128_t lo = wasm_f32x4_splat(0.0f);  // lane[0..3]
  v128_t hi = wasm_f32x4_splat(0.0f);  // lane[4..7]
};

// The scalar reduce_lanes, instruction for instruction:
//   half=4:  lane[i] += lane[i + 4]   for i in 0..3   -> lo + hi
//   half=2:  lane[i] += lane[i + 2]   for i in 0..1
//   half=1:  lane[0] += lane[1]
// Each step adds the same two values the scalar loop adds, so the rounding at
// every node of the tree is the same. The lanes a shuffle leaves as don't-care
// are never read.
inline float reduce_lanes(const Acc& acc) {
  const v128_t s4 = wasm_f32x4_add(acc.lo, acc.hi);                        // half = 4
  const v128_t s2 = wasm_f32x4_add(s4, wasm_i32x4_shuffle(s4, s4, 2, 3, 2, 3));  // half = 2
  const v128_t s1 = wasm_f32x4_add(s2, wasm_i32x4_shuffle(s2, s2, 1, 1, 1, 1));  // half = 1
  return wasm_f32x4_extract_lane(s1, 0);
}

// The tail is at most seven elements and joins the lanes it lines up with,
// exactly as the scalar backend does. Scalar-side, as in the AVX2 backend, to
// keep that correspondence obvious; the engine's shapes rarely have a tail.
inline float finish_tail(Acc acc, const float* a, const float* b, size_t body,
                         size_t tail) {
  if (tail == 0) return reduce_lanes(acc);
  float lane[kAccumLanes];
  wasm_v128_store(lane, acc.lo);
  wasm_v128_store(lane + 4, acc.hi);
  for (size_t i = 0; i < tail; ++i) {
    lane[i] += (b == nullptr) ? a[body + i] : a[body + i] * b[body + i];
  }
  acc.lo = wasm_v128_load(lane);
  acc.hi = wasm_v128_load(lane + 4);
  return reduce_lanes(acc);
}

// Widen eight int8 to eight fp32, as two registers of four. Every conversion
// is exact -- each int8 value is representable in i16, i32 and fp32 -- so this
// introduces no rounding and the multiply after it is the scalar backend's.
struct Widened {
  v128_t lo;
  v128_t hi;
};

inline Widened widen_i8(const int8_t* q) {
  const v128_t h = wasm_i16x8_load8x8(q);  // eight int8 -> eight i16
  return {wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(h)),
          wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(h))};
}

}  // namespace

float dot(const float* a, const float* b, size_t n) {
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  Acc acc;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    // Multiply, then add. Two roundings, as in the scalar backend.
    acc.lo = wasm_f32x4_add(acc.lo, wasm_f32x4_mul(wasm_v128_load(a + i),
                                                   wasm_v128_load(b + i)));
    acc.hi = wasm_f32x4_add(acc.hi, wasm_f32x4_mul(wasm_v128_load(a + i + 4),
                                                   wasm_v128_load(b + i + 4)));
  }
  return finish_tail(acc, a, b, body, tail);
}

float dot_i8(const float* a, const int8_t* q, size_t n) {
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  Acc acc;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    const Widened w = widen_i8(q + i);
    acc.lo = wasm_f32x4_add(acc.lo, wasm_f32x4_mul(wasm_v128_load(a + i), w.lo));
    acc.hi = wasm_f32x4_add(acc.hi, wasm_f32x4_mul(wasm_v128_load(a + i + 4), w.hi));
  }
  if (tail == 0) return reduce_lanes(acc);
  float lane[kAccumLanes];
  wasm_v128_store(lane, acc.lo);
  wasm_v128_store(lane + 4, acc.hi);
  for (size_t i = 0; i < tail; ++i) {
    lane[i] += a[body + i] * static_cast<float>(q[body + i]);
  }
  acc.lo = wasm_v128_load(lane);
  acc.hi = wasm_v128_load(lane + 4);
  return reduce_lanes(acc);
}

void axpy(float alpha, const float* x, float* y, size_t n) {
  // Elementwise, so there is no summation order to preserve -- only the
  // requirement that each y[i] is a rounded product added to a rounded value.
  const v128_t va = wasm_f32x4_splat(alpha);
  const size_t tail = n % 4;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += 4) {
    wasm_v128_store(y + i, wasm_f32x4_add(wasm_v128_load(y + i),
                                          wasm_f32x4_mul(va, wasm_v128_load(x + i))));
  }
  for (size_t i = body; i < n; ++i) y[i] += alpha * x[i];
}

void axpy_i8(float alpha, const int8_t* q, float* y, size_t n) {
  const v128_t va = wasm_f32x4_splat(alpha);
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    const Widened w = widen_i8(q + i);
    wasm_v128_store(y + i, wasm_f32x4_add(wasm_v128_load(y + i),
                                          wasm_f32x4_mul(va, w.lo)));
    wasm_v128_store(y + i + 4, wasm_f32x4_add(wasm_v128_load(y + i + 4),
                                              wasm_f32x4_mul(va, w.hi)));
  }
  for (size_t i = body; i < n; ++i) y[i] += alpha * static_cast<float>(q[i]);
}

float sum(const float* x, size_t n) {
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  Acc acc;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    acc.lo = wasm_f32x4_add(acc.lo, wasm_v128_load(x + i));
    acc.hi = wasm_f32x4_add(acc.hi, wasm_v128_load(x + i + 4));
  }
  return finish_tail(acc, x, nullptr, body, tail);
}

float max(const float* x, size_t n) {
  // Left scalar on purpose, as in every backend. f32x4.max propagates NaN and
  // orders -0 below +0; `>` does neither. Softmax calls this once per row.
  float m = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < n; ++i) {
    if (x[i] > m) m = x[i];
  }
  return m;
}

void scale(float* x, float alpha, size_t n) {
  const v128_t va = wasm_f32x4_splat(alpha);
  const size_t tail = n % 4;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += 4) {
    wasm_v128_store(x + i, wasm_f32x4_mul(wasm_v128_load(x + i), va));
  }
  for (size_t i = body; i < n; ++i) x[i] *= alpha;
}

void add(const float* x, float* y, size_t n) {
  const size_t tail = n % 4;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += 4) {
    wasm_v128_store(y + i, wasm_f32x4_add(wasm_v128_load(y + i),
                                          wasm_v128_load(x + i)));
  }
  for (size_t i = body; i < n; ++i) y[i] += x[i];
}

}  // namespace gpt2::backend
