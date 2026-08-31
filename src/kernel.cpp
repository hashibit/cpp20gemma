#include "kernel.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace gemma {

// ---------------------------------------------------------------------------
// int8 x f32 dot product, scale applied exactly once at the end
// ---------------------------------------------------------------------------
float dot_i8_f32(const int8_t* w, const float* x, size_t n, float scale) {
#if defined(__ARM_NEON)
    // 8 elements per iteration = 2x float32x4. Two independent accumulators
    // break the FMLA dependency chain (same two-accumulator shape as the AVX2 path).
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t w8 = vld1_s8(w + i);
        int16x8_t w16 = vmovl_s8(w8);
        float32x4_t wf0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16)));
        float32x4_t wf1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16)));
        float32x4_t xf0 = vld1q_f32(x + i);
        float32x4_t xf1 = vld1q_f32(x + i + 4);
        acc0 = vfmaq_f32(acc0, xf0, wf0);
        acc1 = vfmaq_f32(acc1, xf1, wf1);
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += (float)w[i] * x[i];
    return sum * scale;
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 xv = _mm256_loadu_ps(x + i);
        __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + i));
        __m256i w16 = _mm256_cvtepi8_epi16(w8);
        __m256 wf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(w16));
        acc = _mm256_fmadd_ps(xv, wf, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) sum += (float)w[i] * x[i];
    return sum * scale;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += (float)w[i] * x[i];
    return sum * scale;
#endif
}

// ---------------------------------------------------------------------------
// fp32 x fp32 dot product
// ---------------------------------------------------------------------------
float dot_f32_f32(const float* w, const float* x, size_t n) {
#if defined(__ARM_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(x + i), vld1q_f32(w + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(x + i + 4), vld1q_f32(w + i + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(x + i), _mm256_loadu_ps(w + i), acc);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;
#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) sum += w[i] * x[i];
    return sum;
#endif
}

// ---------------------------------------------------------------------------
// Row dequantization
// ---------------------------------------------------------------------------
void dequant_row(const int8_t* w, float scale, float* out, size_t n) {
#if defined(__ARM_NEON)
    float32x4_t sv = vdupq_n_f32(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        int8x8_t w8 = vld1_s8(w + i);
        int16x8_t w16 = vmovl_s8(w8);
        vst1q_f32(out + i,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w16))), sv));
        vst1q_f32(out + i + 4,
                  vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(w16))), sv));
    }
    for (; i < n; i++) out[i] = (float)w[i] * scale;
#elif defined(__AVX2__)
    __m256 sv = _mm256_set1_ps(scale);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i w8 = _mm_loadl_epi64((const __m128i*)(w + i));
        __m256i w16 = _mm256_cvtepi8_epi16(w8);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(w16)), sv));
    }
    for (; i < n; i++) out[i] = (float)w[i] * scale;
#else
    for (size_t i = 0; i < n; i++) out[i] = (float)w[i] * scale;
#endif
}

// ---------------------------------------------------------------------------
// GEMV / small GEMM over quantized weights
// ---------------------------------------------------------------------------
void gemv_i8_f32(const QTensor& w, const float* x, float* out) {
    for (uint32_t r = 0; r < w.rows; r++)
        out[r] = dot_i8_f32(w.data + (size_t)r * w.cols, x, w.cols, w.scale);
}

void gemm_i8_f32(const QTensor& w, const float* x, size_t m, float* out) {
    for (size_t i = 0; i < m; i++)
        for (uint32_t r = 0; r < w.rows; r++)
            out[i * w.rows + r] = dot_i8_f32(w.data + (size_t)r * w.cols, x + i * w.cols, w.cols, w.scale);
}

}  // namespace gemma
