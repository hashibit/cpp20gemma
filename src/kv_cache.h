// Append-only KV cache: one preallocated block per layer.
// The cache stores rope-applied k (applied before the write — see model.cpp).
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace gemma {

struct KVCache {
    uint32_t max_len = 0;  // allocated positions
    uint32_t kv_dim = 0;   // n_kv_heads * head_dim
    std::vector<float> k;  // [max_len * kv_dim]
    std::vector<float> v;

    void init(uint32_t max_len_, uint32_t kv_dim_) {
        max_len = max_len_;
        kv_dim = kv_dim_;
        k.assign((size_t)max_len * kv_dim, 0.0f);
        v.assign((size_t)max_len * kv_dim, 0.0f);
    }

    void write(uint32_t pos, const float* k_in, const float* v_in) {
        std::memcpy(k.data() + (size_t)pos * kv_dim, k_in, kv_dim * sizeof(float));
        std::memcpy(v.data() + (size_t)pos * kv_dim, v_in, kv_dim * sizeof(float));
    }

    const float* k_row(uint32_t pos) const { return k.data() + (size_t)pos * kv_dim; }
    const float* v_row(uint32_t pos) const { return v.data() + (size_t)pos * kv_dim; }
};

}  // namespace gemma
