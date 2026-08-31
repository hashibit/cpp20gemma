// Forward pass: embedding -> transformer blocks -> final norm -> logits.
//
// Architecture:
//   hidden = embed_tokens[tok] + embed_positions[pos]
//   per layer:  h = h + attn(rmsnorm(h)) ;  h = h + mlp(rmsnorm(h))
//   logits = lm_head(rmsnorm(h))
// attn = GQA with RoPE (rope-applied k cached), mlp = GeGLU.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "config.h"
#include "kv_cache.h"
#include "ops.h"
#include "weights.h"

namespace gemma {

// --- layer blocks, exposed as free functions so unit tests can drive them ---

// x: [dim] post-attn-norm hidden; heads_out: [n_heads*head_dim] concat heads;
// out: [dim] o_proj result. q/k/v/scores are scratch (see Model::alloc).
void attention_block(const Weights& w, const ModelConfig& cfg, uint32_t layer,
                     const float* x, uint32_t pos, KVCache& kv, RopeCache& rope,
                     float* heads_out, float* out,
                     float* q_buf, float* k_buf, float* v_buf, float* scores);

// x: [dim] post-ffn-norm hidden; out: [dim]; gate/up are [mlp_dim] scratch.
void mlp_block(const Weights& w, const ModelConfig& cfg, uint32_t layer,
               const float* x, float* out, float* gate_buf, float* up_buf);

// ---------------------------------------------------------------------------
struct Model {
    const Weights& w;
    ModelConfig cfg;
    uint32_t n_ctx = 0;  // effective max positions (cache cap)

    std::vector<KVCache> kvc;  // [n_layers]
    RopeCache rope;

    // Activation workspace — preallocated once, zero allocations per step.
    std::vector<float> hidden;    // [dim]
    std::vector<float> h_norm;    // [dim]
    std::vector<float> pos_emb;   // [dim]
    std::vector<float> q;         // [n_heads*head_dim]
    std::vector<float> k, v;      // [n_kv_heads*head_dim]
    std::vector<float> heads;     // [n_heads*head_dim]
    std::vector<float> attn_out;  // [dim]
    std::vector<float> ffn_norm;  // [dim]
    std::vector<float> gate, up;  // [mlp_dim]
    std::vector<float> mlp_out;   // [dim]
    std::vector<float> scores;    // [n_ctx]
    std::vector<float> logits;    // [vocab_size]

    bool pos_warned = false;

    Model(const Weights& weights, uint32_t n_ctx_);

    // One forward step at absolute position `pos`; fills logits.
    void forward_token(uint32_t token, uint32_t pos);

    std::span<const float> logits_span() const { return logits; }
};

}  // namespace gemma
