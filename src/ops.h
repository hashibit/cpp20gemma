// Elementwise + reduction ops: RMSNorm, RoPE, SiLU/GeGLU, softmax.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gemma {

// out[i] = x[i] / sqrt(mean(x^2) + eps) * gamma[i]
void rmsnorm(const float* x, const float* gamma, size_t n, float eps, float* out);

// ---------------------------------------------------------------------------
// RoPE with a lazily materialized per-position cos/sin table.
// freqs[j] = base^(-2j/head_dim), j in [0, head_dim/2). GPT-NeoX interleaved
// pairs: x'[2j]   = x[2j]*cos - x[2j+1]*sin
//         x'[2j+1] = x[2j]*sin + x[2j+1]*cos
// ---------------------------------------------------------------------------
struct RopeCache {
    std::vector<float> cos_t;  // [pos][head_dim/2]
    std::vector<float> sin_t;  // [pos][head_dim/2]
    uint32_t head_dim = 0;
    uint32_t nhalf = 0;
    uint32_t max_pos = 0;      // materialized rows
    float base = 10000.0f;

    RopeCache() = default;
    RopeCache(uint32_t head_dim, float rope_base);

    // Materialize cos/sin rows up to and including `pos`.
    void ensure(uint32_t pos);

    // Apply rotation to x[0..head_dim) at absolute position pos (in place).
    void apply(float* x, uint32_t pos) const;
};

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------
float silu(float x);
// out[i] = x[i] * sigmoid(x[i])
void silu_inplace(float* x, size_t n);
// GeGLU: gate[i] = silu(gate[i]) * up[i]
void geglu_inplace(float* gate, const float* up, size_t n);

// In-place numerically-stable softmax over x[0..n).
void softmax(float* x, size_t n);

}  // namespace gemma
