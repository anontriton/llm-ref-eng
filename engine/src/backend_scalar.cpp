// Scalar reference backend.
//
// Straight-line fp32, no intrinsics, no unrolling that changes summation order.
// This is the backend Phase 2 is validated on: it is the one whose numerics are
// easiest to reason about, so when the oracle disagrees the kernel is at fault
// rather than the arithmetic.
#include "gpt2/backend/backend.h"

#include <limits>

namespace gpt2::backend {

const char* name() { return "scalar"; }

namespace {

// Reduce the lanes pairwise rather than left to right. With kAccumLanes a power
// of two this is a fixed, shallow tree, so it is both deterministic and about
// as accurate as the lanes allow.
float reduce_lanes(float* lane) {
  for (size_t half = kAccumLanes / 2; half > 0; half /= 2) {
    for (size_t i = 0; i < half; ++i) lane[i] += lane[i + half];
  }
  return lane[0];
}

}  // namespace

float dot(const float* a, const float* b, size_t n) {
  float lane[kAccumLanes] = {0.0f};
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    for (size_t k = 0; k < kAccumLanes; ++k) lane[k] += a[i + k] * b[i + k];
  }
  // The tail joins the lanes it lines up with, so a length that is not a
  // multiple of the lane count still has one well-defined answer.
  for (size_t i = 0; i < tail; ++i) lane[i] += a[body + i] * b[body + i];
  return reduce_lanes(lane);
}

void axpy(float alpha, const float* x, float* y, size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

float sum(const float* x, size_t n) {
  float lane[kAccumLanes] = {0.0f};
  const size_t tail = n % kAccumLanes;
  const size_t body = n - tail;
  for (size_t i = 0; i < body; i += kAccumLanes) {
    for (size_t k = 0; k < kAccumLanes; ++k) lane[k] += x[i + k];
  }
  for (size_t i = 0; i < tail; ++i) lane[i] += x[body + i];
  return reduce_lanes(lane);
}

float max(const float* x, size_t n) {
  float m = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < n; ++i) {
    if (x[i] > m) m = x[i];
  }
  return m;
}

void scale(float* x, float alpha, size_t n) {
  for (size_t i = 0; i < n; ++i) x[i] *= alpha;
}

void add(const float* x, float* y, size_t n) {
  for (size_t i = 0; i < n; ++i) y[i] += x[i];
}

}  // namespace gpt2::backend
