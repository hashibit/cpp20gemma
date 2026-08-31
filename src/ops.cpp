#include "ops.h"

#include <cmath>
#include <cstring>

namespace gemma {

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------
void rmsnorm(const float* x, const float* gamma, size_t n, float eps, float* out) {
    // mean(x^2) — sequential fp32 accumulation (mirrors the NumPy reference).
    float ss = 0.0f;
    for (size_t i = 0; i < n; i++) ss += x[i] * x[i];
    float rstd = 1.0f / std::sqrt(ss / (float)n + eps);
    for (size_t i = 0; i < n; i++) out[i] = x[i] * rstd * gamma[i];
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------
RopeCache::RopeCache(uint32_t head_dim_, float rope_base)
    : head_dim(head_dim_), nhalf(head_dim_ / 2), base(rope_base) {}

void RopeCache::ensure(uint32_t pos) {
    if (pos < max_pos) return;
    const uint32_t new_rows = pos + 1;
    cos_t.resize((size_t)new_rows * nhalf);
    sin_t.resize((size_t)new_rows * nhalf);
    // freqs[j] = base^(-2j/head_dim)
    std::vector<float> freqs(nhalf);
    for (uint32_t j = 0; j < nhalf; j++)
        freqs[j] = 1.0f / std::pow(base, 2.0f * (float)j / (float)head_dim);
    for (uint32_t p = max_pos; p < new_rows; p++) {
        for (uint32_t j = 0; j < nhalf; j++) {
            float a = (float)p * freqs[j];
            cos_t[(size_t)p * nhalf + j] = std::cos(a);
            sin_t[(size_t)p * nhalf + j] = std::sin(a);
        }
    }
    max_pos = new_rows;
}

void RopeCache::apply(float* x, uint32_t pos) const {
    const float* c = cos_t.data() + (size_t)pos * nhalf;
    const float* s = sin_t.data() + (size_t)pos * nhalf;
    for (uint32_t j = 0; j < nhalf; j++) {
        float x0 = x[2 * j], x1 = x[2 * j + 1];
        x[2 * j]     = x0 * c[j] - x1 * s[j];
        x[2 * j + 1] = x0 * s[j] + x1 * c[j];
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------
float silu(float x) {
    // x * sigmoid(x); computed as x / (1 + exp(-x)) is less accurate for
    // large negative x, so use the exp form with the standard overflow guard.
    float e = std::exp(x);
    return x * e / (1.0f + e);
}

void silu_inplace(float* x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = silu(x[i]);
}

void geglu_inplace(float* gate, const float* up, size_t n) {
    for (size_t i = 0; i < n; i++) gate[i] = silu(gate[i]) * up[i];
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------
void softmax(float* x, size_t n) {
    float mx = x[0];
    for (size_t i = 1; i < n; i++)
        if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        x[i] = std::exp(x[i] - mx);
        sum += x[i];
    }
    for (size_t i = 0; i < n; i++) x[i] /= sum;
}

}  // namespace gemma
