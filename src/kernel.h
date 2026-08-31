// SIMD kernels: int8×f32 dot, f32×f32 dot, row dequantization.
//
// Three compile-time paths sharing one algorithm:
//   - NEON  (arm64, this Mac): int8 -> int16 -> int32 -> f32 widen + FMLA
//   - AVX2  (x86-64): _mm256_cvtepi8_epi16 + _mm256_fmadd_ps
//   - scalar fallback everywhere else
//
// Key trick: compute the dot product against the raw int8
// weights, then multiply the single accumulated sum by the dequant scale once.
#pragma once

#include <cstddef>
#include <cstdint>

#include "tensor.h"

namespace gemma {

// dot(x, dequant(w)) — w is int8 with per-tensor scale; result = sum * scale.
float dot_i8_f32(const int8_t* w, const float* x, size_t n, float scale);

// Plain fp32 dot product (attention scores, unit-test reference).
float dot_f32_f32(const float* w, const float* x, size_t n);

// out[i] = (float)w[i] * scale for i in [0, n) — embedding row dequantization.
void dequant_row(const int8_t* w, float scale, float* out, size_t n);

// GEMV over a quantized weight matrix: out[i] = dot(row_i, x) * scale.
// w is stored [rows][cols] row-major (each row contiguous).
void gemv_i8_f32(const QTensor& w, const float* x, float* out);

// Small GEMM for prefill batching: x is [m][k] row-major, out is [m][rows].
void gemm_i8_f32(const QTensor& w, const float* x, size_t m, float* out);

}  // namespace gemma
