#include "model.h"

#include <algorithm>
#include <cmath>

#include "kernel.h"

namespace gemma {

// ---------------------------------------------------------------------------
// Attention block (GQA + RoPE + KV cache)
// ---------------------------------------------------------------------------
void attention_block(const Weights& w, const ModelConfig& cfg, uint32_t layer,
                     const float* x, uint32_t pos, KVCache& kv, RopeCache& rope,
                     float* heads_out, float* out,
                     float* q_buf, float* k_buf, float* v_buf, float* scores) {
    const uint32_t hd = cfg.head_dim, nh = cfg.n_heads, nkv = cfg.n_kv_heads;
    const Weights::Layer& L = w.layers[layer];

    gemv_i8_f32(L.q, x, q_buf);
    gemv_i8_f32(L.k, x, k_buf);
    gemv_i8_f32(L.v, x, v_buf);

    for (uint32_t h = 0; h < nh; h++) rope.apply(q_buf + (size_t)h * hd, pos);
    for (uint32_t h = 0; h < nkv; h++) rope.apply(k_buf + (size_t)h * hd, pos);

    // The cache stores rope-applied k (decided once; the NumPy reference
    // mirrors this) so decode steps skip re-rotating history.
    kv.write(pos, k_buf, v_buf);

    const float score_scale = 1.0f / std::sqrt((float)hd);
    for (uint32_t h = 0; h < nh; h++) {
        const uint32_t kvh = h * nkv / nh;
        const float* qh = q_buf + (size_t)h * hd;
        for (uint32_t t = 0; t <= pos; t++)
            scores[t] = dot_f32_f32(qh, kv.k_row(t) + (size_t)kvh * hd, hd) * score_scale;
        softmax(scores, (size_t)pos + 1);

        float* oh = heads_out + (size_t)h * hd;
        for (uint32_t i = 0; i < hd; i++) oh[i] = 0.0f;
        for (uint32_t t = 0; t <= pos; t++) {
            const float* vrow = kv.v_row(t) + (size_t)kvh * hd;
            const float s = scores[t];
            for (uint32_t i = 0; i < hd; i++) oh[i] += s * vrow[i];
        }
    }

    gemv_i8_f32(L.o, heads_out, out);
}

// ---------------------------------------------------------------------------
// MLP block (GeGLU)
// ---------------------------------------------------------------------------
void mlp_block(const Weights& w, const ModelConfig& cfg, uint32_t layer,
               const float* x, float* out, float* gate_buf, float* up_buf) {
    const Weights::Layer& L = w.layers[layer];
    gemv_i8_f32(L.gate, x, gate_buf);
    gemv_i8_f32(L.up, x, up_buf);
    geglu_inplace(gate_buf, up_buf, cfg.mlp_dim);
    gemv_i8_f32(L.down, gate_buf, out);
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------
Model::Model(const Weights& weights, uint32_t n_ctx_) : w(weights) {
    cfg = weights.cfg;
    n_ctx = std::min(n_ctx_, cfg.max_seq_len);

    kvc.resize(cfg.n_layers);
    const uint32_t kv_dim = cfg.n_kv_heads * cfg.head_dim;
    for (auto& c : kvc) c.init(n_ctx, kv_dim);

    rope = RopeCache(cfg.head_dim, cfg.rope_base);

    hidden.assign(cfg.dim, 0.0f);
    h_norm.assign(cfg.dim, 0.0f);
    pos_emb.assign(cfg.dim, 0.0f);
    q.assign((size_t)cfg.n_heads * cfg.head_dim, 0.0f);
    k.assign((size_t)cfg.n_kv_heads * cfg.head_dim, 0.0f);
    v.assign((size_t)cfg.n_kv_heads * cfg.head_dim, 0.0f);
    heads.assign((size_t)cfg.n_heads * cfg.head_dim, 0.0f);
    attn_out.assign(cfg.dim, 0.0f);
    ffn_norm.assign(cfg.dim, 0.0f);
    gate.assign(cfg.mlp_dim, 0.0f);
    up.assign(cfg.mlp_dim, 0.0f);
    mlp_out.assign(cfg.dim, 0.0f);
    scores.assign(n_ctx, 0.0f);
    logits.assign(cfg.vocab_size, 0.0f);
}

void Model::forward_token(uint32_t token, uint32_t pos) {
    if (token >= cfg.vocab_size) {
        if (!pos_warned) {
            warn("token id out of vocab range; using id 0");
            pos_warned = true;
        }
        token = 0;
    }

    // Embedding: token row + learnable position bias (clamped to the table).
    uint32_t p = std::min(pos, w.embed_positions.rows - 1);
    if (p != pos && !pos_warned) {
        warn("position exceeds embed_positions rows; clamping");
        pos_warned = true;
    }
    dequant_row(w.embed_tokens.data + (size_t)token * cfg.dim, w.embed_tokens.scale,
                hidden.data(), cfg.dim);
    dequant_row(w.embed_positions.data + (size_t)p * cfg.dim, w.embed_positions.scale,
                pos_emb.data(), cfg.dim);
    for (uint32_t i = 0; i < cfg.dim; i++) hidden[i] += pos_emb[i];

    rope.ensure(pos);

    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        rmsnorm(hidden.data(), w.gamma_attn_f32[l].data(), cfg.dim, ModelConfig::kRmsEps, h_norm.data());
        attention_block(w, cfg, l, h_norm.data(), pos, kvc[l], rope,
                        heads.data(), attn_out.data(), q.data(), k.data(), v.data(), scores.data());
        for (uint32_t i = 0; i < cfg.dim; i++) hidden[i] += attn_out[i];

        rmsnorm(hidden.data(), w.gamma_ffn_f32[l].data(), cfg.dim, ModelConfig::kRmsEps, ffn_norm.data());
        mlp_block(w, cfg, l, ffn_norm.data(), mlp_out.data(), gate.data(), up.data());
        for (uint32_t i = 0; i < cfg.dim; i++) hidden[i] += mlp_out[i];
    }

    rmsnorm(hidden.data(), w.final_norm_gamma_f32.data(), cfg.dim, ModelConfig::kRmsEps, h_norm.data());
    gemv_i8_f32(w.lm_head, h_norm.data(), logits.data());
}

}  // namespace gemma
