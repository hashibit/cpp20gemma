// Weights file loader (GMW1 format): one big aligned buffer + descriptors.
#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

#include "config.h"
#include "tensor.h"

namespace gemma {

struct Weights {
    ModelConfig cfg;
    AlignedBuffer buf;  // whole file, 64B-aligned

    QTensor embed_tokens;     // [vocab][dim]
    QTensor embed_positions;  // [max_seq_len][dim]
    struct Layer {
        QTensor q, k, v, o;    // [dim][dim], stored [out][in] (transposed)
        QTensor gate, up;      // [mlp_dim][dim]
        QTensor down;          // [dim][mlp_dim], stored [out][in]
        QTensor gamma_attn;    // [dim]
        QTensor gamma_ffn;     // [dim]
    };
    std::vector<Layer> layers;
    QTensor final_norm_gamma;  // [dim]
    QTensor lm_head;           // [vocab][dim], stored [out][in]

    // Norm gammas dequantized once at load (they are hit every forward pass).
    std::vector<std::vector<float>> gamma_attn_f32;  // [n_layers][dim]
    std::vector<std::vector<float>> gamma_ffn_f32;
    std::vector<float> final_norm_gamma_f32;

    bool load(const char* path);
    void dump(FILE* f) const;

    // Exact file size implied by the header (shape validation for free).
    static size_t expected_file_size(const ModelConfig& cfg);
};

}  // namespace gemma
