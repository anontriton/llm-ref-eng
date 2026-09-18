// The SIMD seam.
//
// Kernel logic in src/ops.cpp is written against these primitives and contains
// no intrinsics of its own. Phase 3 swaps in an AVX2 implementation and Phase 5
// a wasm_simd128 one by providing another translation unit that defines these
// same functions; the kernels above do not change.
//
// The seam is deliberately at the level of whole-array reductions and sweeps
// rather than individual vector registers, because that is the level at which
// AVX2 and wasm_simd128 differ (256-bit vs 128-bit) without the kernel caring.
//
// Numerics contract, which matters more than speed until Phase 3:
//   - dot() and sum() accumulate in fp32 across exactly kAccumLanes
//     independent lanes, then reduce those lanes pairwise. The lane count is
//     fixed here rather than chosen per backend so that scalar, AVX2 and
//     wasm_simd128 produce bit-identical results: a backend swap must not move
//     the numbers, or oracle validation stops meaning anything across Phase 3
//     and Phase 5.
//   - No fused multiply-add is assumed. FMA changes results by withholding an
//     intermediate rounding; that is a Phase 3 decision to make deliberately,
//     not something a compiler flag should introduce behind our back.
//
// Why lanes at all: a single sequential fp32 accumulator over a 3072-deep dot
// product loses roughly n*eps, which measured ~10x worse than the PyTorch
// reference's blocked GEMM and put the residual stream outside the 1e-4
// tolerance. Splitting the chain into lanes cuts that to ~(n/lanes + lanes).
#pragma once

#include <cstddef>

namespace gpt2::backend {

// Independent fp32 accumulator lanes used by every reduction. Eight because
// that is one AVX2 register of floats; wasm_simd128 holds four per register and
// so spends two, which costs an instruction but keeps the arithmetic identical.
inline constexpr size_t kAccumLanes = 8;

// Name of the active backend, for benchmark JSON and dump manifests.
const char* name();

// sum_i a[i] * b[i]
float dot(const float* a, const float* b, size_t n);

// y[i] += alpha * x[i]
void axpy(float alpha, const float* x, float* y, size_t n);

// sum_i x[i]
float sum(const float* x, size_t n);

// max_i x[i]. Returns -inf for n == 0.
float max(const float* x, size_t n);

// x[i] *= alpha
void scale(float* x, float alpha, size_t n);

// y[i] = x[i] + y[i]
void add(const float* x, float* y, size_t n);

}  // namespace gpt2::backend
