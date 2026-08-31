// Unit tests: every layer and kernel compared against NumPy golden data
// (see py/generate_golden.py). Self-contained TEST/CHECK macro harness —
// exit code is the failure count.
//
// Usage: ./build/unit_tests tests/golden weights/tiny_weights.bin weights/tiny_tokenizer.bin
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "config.h"
#include "kernel.h"
#include "kv_cache.h"
#include "model.h"
#include "ops.h"
#include "sampler.h"
#include "tokenizer.h"
#include "weights.h"

using namespace gemma;

static int g_fail = 0;
static int g_count = 0;
static std::string g_golden, g_weights_path, g_tok_path;

static const uint32_t PROMPT_TOKS[8] = {13, 7, 42, 99, 5, 21, 8, 3};

#define TEST(name) void test_##name()
#define RUN(name)                      \
    do {                               \
        std::printf("== %s\n", #name); \
        test_##name();                 \
        g_count++;                     \
    } while (0)

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                  \
            g_fail++;                                                                      \
        }                                                                                  \
    } while (0)

#define CHECK_EQ(a, b)                                                                                  \
    do {                                                                                                \
        auto va = (a);                                                                                  \
        auto vb = (b);                                                                                  \
        if (!(va == vb)) {                                                                              \
            std::printf("  FAIL %s:%d: %s == %s (%lld vs %lld)\n", __FILE__, __LINE__, #a, #b,          \
                        (long long)va, (long long)vb);                                                  \
            g_fail++;                                                                                   \
        }                                                                                               \
    } while (0)

#define CHECK_NEAR(a, b, atol, rtol)                                                                   \
    do {                                                                                                \
        float va = (float)(a), vb = (float)(b);                                                         \
        float d = std::fabs(va - vb);                                                                   \
        if (!(d <= (atol) + (rtol) * std::fabs(vb))) {                                                  \
            std::printf("  FAIL %s:%d: |%s - %s| = %g (got %g, want %g)\n", __FILE__, __LINE__, #a, #b,  \
                        d, va, vb);                                                                     \
            g_fail++;                                                                                   \
        }                                                                                               \
    } while (0)

// --- file helpers ----------------------------------------------------------
static std::vector<char> read_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    CHECK(f != nullptr);
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> v((size_t)n);
    if (n > 0) std::fread(v.data(), 1, (size_t)n, f);
    std::fclose(f);
    return v;
}

static std::vector<float> read_f32(const std::string& path) {
    std::vector<char> b = read_file(path);
    std::vector<float> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

static std::vector<uint32_t> read_u32(const std::string& path) {
    std::vector<char> b = read_file(path);
    std::vector<uint32_t> v(b.size() / 4);
    std::memcpy(v.data(), b.data(), v.size() * 4);
    return v;
}

static void write_file(const std::string& path, const std::vector<char>& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (!f) return;
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

// --- tests -----------------------------------------------------------------
TEST(weights_loader) {
    Weights w;
    CHECK(w.load(g_weights_path.c_str()));
    CHECK_EQ(w.cfg.n_layers, 2u);
    CHECK_EQ(w.cfg.dim, 256u);
    CHECK_EQ(w.cfg.n_heads, 4u);
    CHECK_EQ(w.cfg.n_kv_heads, 1u);
    CHECK_EQ(w.cfg.mlp_dim, 1024u);
    CHECK_EQ(w.cfg.vocab_size, 1024u);
    CHECK_EQ(w.cfg.max_seq_len, 256u);
    CHECK_EQ(w.cfg.head_dim, 64u);

    std::vector<char> good = read_file(g_weights_path);
    CHECK(good.size() > 64);
    // file size must match the header-implied size exactly; the aligned
    // buffer is only >= that (rounded up to 64 bytes)
    CHECK_EQ(good.size(), Weights::expected_file_size(w.cfg));
    CHECK(w.buf.size >= Weights::expected_file_size(w.cfg));

    auto bad = good;
    bad[0] = 'X';
    write_file("/tmp/gemma_bad_magic.bin", bad);
    Weights w1;
    CHECK(!w1.load("/tmp/gemma_bad_magic.bin"));

    bad = good;
    bad[4] = 9;  // version
    write_file("/tmp/gemma_bad_version.bin", bad);
    Weights w2;
    CHECK(!w2.load("/tmp/gemma_bad_version.bin"));

    bad = good;
    bad.resize(bad.size() - 10);  // truncated
    write_file("/tmp/gemma_truncated.bin", bad);
    Weights w3;
    CHECK(!w3.load("/tmp/gemma_truncated.bin"));

    bad = good;
    bad.push_back(0);  // size mismatch (shape validation)
    write_file("/tmp/gemma_wrong_size.bin", bad);
    Weights w4;
    CHECK(!w4.load("/tmp/gemma_wrong_size.bin"));
}

TEST(matmul_i8) {
    std::vector<char> f = read_file(g_golden + "/mm_i8.bin");
    CHECK_EQ(f.size(), 16 * 32 + 4 + 32 * 4 + 16 * 4);
    const int8_t* W = (const int8_t*)f.data();
    float scale;
    std::memcpy(&scale, f.data() + 16 * 32, 4);
    const float* x = (const float*)(f.data() + 16 * 32 + 4);
    const float* y = (const float*)(f.data() + 16 * 32 + 4 + 32 * 4);

    QTensor t{W, scale, 16, 32};
    std::vector<float> out(16);
    gemv_i8_f32(t, x, out.data());
    for (int i = 0; i < 16; i++) CHECK_NEAR(out[i], y[i], 2e-3f, 1e-3f);

    // scale is applied exactly once: dot * scale == sum((int8)x) * scale
    CHECK_NEAR(dot_i8_f32(W, x, 32, scale), y[0], 2e-3f, 1e-3f);

    // zero-weight guard: all-zero tensor must produce exactly 0, not NaN
    int8_t zw[64] = {};
    float zx[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(dot_i8_f32(zw, zx, 8, 1e-9f / 127.0f) == 0.0f);

    // tail handling: odd length == NEON main loop + 1 scalar
    float tx[9] = {1, -2, 3, -4, 5, -6, 7, -8, 9};
    int8_t tw[9] = {2, 3, -4, 5, -6, 7, -8, 9, 10};
    float ref = 0;
    for (int i = 0; i < 9; i++) ref += (float)tw[i] * tx[i];
    CHECK_NEAR(dot_i8_f32(tw, tx, 9, 1.0f), ref, 1e-5f, 1e-6f);
}

TEST(matmul_f32) {
    std::vector<char> f = read_file(g_golden + "/mm_f32.bin");
    CHECK_EQ(f.size(), 16 * 32 * 4 + 32 * 4 + 16 * 4);
    const float* W = (const float*)f.data();
    const float* x = (const float*)(f.data() + 16 * 32 * 4);
    const float* y = (const float*)(f.data() + 16 * 32 * 4 + 32 * 4);
    for (int i = 0; i < 16; i++)
        CHECK_NEAR(dot_f32_f32(W + i * 32, x, 32), y[i], 2e-3f, 1e-3f);
}

TEST(dequant_row_test) {
    Weights w;
    CHECK(w.load(g_weights_path.c_str()));
    auto golden = read_f32(g_golden + "/dequant_row.bin");
    CHECK_EQ(golden.size(), 256u);
    std::vector<float> out(256);
    dequant_row(w.embed_tokens.data + 13 * 256, w.embed_tokens.scale, out.data(), 256);
    for (int i = 0; i < 256; i++) CHECK_EQ(out[i], golden[i]);  // bit-exact by construction
}

TEST(rmsnorm_test) {
    Weights w;
    CHECK(w.load(g_weights_path.c_str()));
    auto emb = read_f32(g_golden + "/embedding_out.bin");
    auto norm_a = read_f32(g_golden + "/norm_a.bin");
    CHECK_EQ(emb.size(), 8u * 256);
    std::vector<float> out(256);
    rmsnorm(emb.data(), w.gamma_attn_f32[0].data(), 256, ModelConfig::kRmsEps, out.data());
    for (int i = 0; i < 256; i++) CHECK_NEAR(out[i], norm_a[i], 2e-3f, 1e-3f);
}

TEST(rope_test) {
    auto in = read_f32(g_golden + "/rope_in.bin");
    auto golden = read_f32(g_golden + "/rope_out.bin");
    CHECK_EQ(in.size(), 64u);

    RopeCache rope(64, 10000.0f);
    rope.ensure(3);
    std::vector<float> x = in;
    rope.apply(x.data(), 3);
    for (int i = 0; i < 64; i++) CHECK_NEAR(x[i], golden[i], 2e-3f, 1e-3f);

    // position 0 is the identity rotation
    x = in;
    rope.apply(x.data(), 0);
    for (int i = 0; i < 64; i++) CHECK_NEAR(x[i], in[i], 1e-5f, 1e-6f);
}

TEST(silu_geglu_test) {
    auto in = read_f32(g_golden + "/silu_in.bin");
    auto golden = read_f32(g_golden + "/silu_out.bin");
    std::vector<float> x = in;
    silu_inplace(x.data(), x.size());
    for (size_t i = 0; i < x.size(); i++) CHECK_NEAR(x[i], golden[i], 1e-4f, 1e-5f);

    auto g = read_f32(g_golden + "/geglu_gate_in.bin");
    auto u = read_f32(g_golden + "/geglu_up_in.bin");
    auto o = read_f32(g_golden + "/geglu_out.bin");
    geglu_inplace(g.data(), u.data(), g.size());
    for (size_t i = 0; i < g.size(); i++) CHECK_NEAR(g[i], o[i], 1e-4f, 1e-5f);
}

TEST(softmax_test) {
    auto in = read_f32(g_golden + "/softmax_in.bin");
    auto golden = read_f32(g_golden + "/softmax_out.bin");
    std::vector<float> x = in;
    softmax(x.data(), x.size());
    float sum = 0;
    for (size_t i = 0; i < x.size(); i++) {
        CHECK_NEAR(x[i], golden[i], 1e-4f, 1e-5f);
        sum += x[i];
    }
    CHECK_NEAR(sum, 1.0f, 1e-5f, 1e-6f);

    // stability with large-magnitude inputs
    auto big_in = read_f32(g_golden + "/softmax_big_in.bin");
    auto big_out = read_f32(g_golden + "/softmax_big_out.bin");
    x = big_in;
    softmax(x.data(), x.size());
    sum = 0;
    for (size_t i = 0; i < x.size(); i++) {
        CHECK_NEAR(x[i], big_out[i], 1e-4f, 1e-5f);
        CHECK(x[i] == x[i]);  // not NaN
        sum += x[i];
    }
    CHECK_NEAR(sum, 1.0f, 1e-5f, 1e-6f);
}

// Full 8-token forward, block by block, compared against per-layer goldens.
TEST(forward_blocks) {
    Weights w;
    CHECK(w.load(g_weights_path.c_str()));
    const ModelConfig& cfg = w.cfg;

    auto emb = read_f32(g_golden + "/embedding_out.bin");        // [8, dim]
    auto norm_a_g = read_f32(g_golden + "/norm_a.bin");          // [L, 8, dim]
    auto attn_g = read_f32(g_golden + "/attn_out.bin");
    auto norm_f_g = read_f32(g_golden + "/norm_f.bin");
    auto mlp_g = read_f32(g_golden + "/mlp_out.bin");
    auto fin_g = read_f32(g_golden + "/final_norm_out.bin");     // [8, dim]
    auto logits_g = read_f32(g_golden + "/logits.bin");          // [8, vocab]

    Model m(w, cfg.max_seq_len);
    const uint32_t d = cfg.dim;
    for (uint32_t pos = 0; pos < 8; pos++) {
        // embedding (mirrors forward_token)
        dequant_row(w.embed_tokens.data + (size_t)PROMPT_TOKS[pos] * d, w.embed_tokens.scale,
                    m.hidden.data(), d);
        dequant_row(w.embed_positions.data + (size_t)pos * d, w.embed_positions.scale,
                    m.pos_emb.data(), d);
        for (uint32_t i = 0; i < d; i++) m.hidden[i] += m.pos_emb[i];
        for (uint32_t i = 0; i < d; i++)
            CHECK_NEAR(m.hidden[i], emb[(size_t)pos * d + i], 2e-3f, 1e-3f);

        m.rope.ensure(pos);
        for (uint32_t l = 0; l < cfg.n_layers; l++) {
            const size_t o = ((size_t)l * 8 + pos) * d;
            rmsnorm(m.hidden.data(), w.gamma_attn_f32[l].data(), d, ModelConfig::kRmsEps, m.h_norm.data());
            for (uint32_t i = 0; i < d; i++) CHECK_NEAR(m.h_norm[i], norm_a_g[o + i], 2e-3f, 1e-3f);

            attention_block(w, cfg, l, m.h_norm.data(), pos, m.kvc[l], m.rope,
                            m.heads.data(), m.attn_out.data(),
                            m.q.data(), m.k.data(), m.v.data(), m.scores.data());
            for (uint32_t i = 0; i < d; i++) CHECK_NEAR(m.attn_out[i], attn_g[o + i], 2e-3f, 3e-3f);  // 点积重抵消处噪声最坏 ~0.2% 相对
            for (uint32_t i = 0; i < d; i++) m.hidden[i] += m.attn_out[i];

            rmsnorm(m.hidden.data(), w.gamma_ffn_f32[l].data(), d, ModelConfig::kRmsEps, m.ffn_norm.data());
            for (uint32_t i = 0; i < d; i++) CHECK_NEAR(m.ffn_norm[i], norm_f_g[o + i], 2e-3f, 1e-3f);

            mlp_block(w, cfg, l, m.ffn_norm.data(), m.mlp_out.data(), m.gate.data(), m.up.data());
            // mlp_out is the largest-magnitude intermediate (1024-length fp32
            // dot of ~O(100) products, plus gate/up projection error amplified
            // through the GeGLU product): accumulation-order noise reaches
            // ~1% relative at heavy-cancellation elements (measured worst
            // 0.49 on |x| ~ 47), so it gets a wider rtol than other layers
            for (uint32_t i = 0; i < d; i++) CHECK_NEAR(m.mlp_out[i], mlp_g[o + i], 2e-3f, 2e-2f);
            for (uint32_t i = 0; i < d; i++) m.hidden[i] += m.mlp_out[i];
        }

        rmsnorm(m.hidden.data(), w.final_norm_gamma_f32.data(), d, ModelConfig::kRmsEps, m.h_norm.data());
        for (uint32_t i = 0; i < d; i++) CHECK_NEAR(m.h_norm[i], fin_g[(size_t)pos * d + i], 2e-3f, 1e-3f);

        gemv_i8_f32(w.lm_head, m.h_norm.data(), m.logits.data());
        for (uint32_t i = 0; i < cfg.vocab_size; i++)
            CHECK_NEAR(m.logits[i], logits_g[(size_t)pos * cfg.vocab_size + i], 2e-3f, 1e-3f);
    }
}

// Raw attention scores, and the rope-before-cache convention on the K path.
TEST(attention_scores_kv) {
    Weights w;
    CHECK(w.load(g_weights_path.c_str()));
    const ModelConfig& cfg = w.cfg;
    auto scores_g = read_f32(g_golden + "/scores_L1.bin");
    auto k_raw_g = read_f32(g_golden + "/k_raw_L1.bin");
    auto k_cached_g = read_f32(g_golden + "/k_cached_L1.bin");

    Model m(w, cfg.max_seq_len);
    const uint32_t d = cfg.dim;
    std::vector<float> xin_l1(d);

    for (uint32_t pos = 0; pos < 8; pos++) {
        dequant_row(w.embed_tokens.data + (size_t)PROMPT_TOKS[pos] * d, w.embed_tokens.scale,
                    m.hidden.data(), d);
        dequant_row(w.embed_positions.data + (size_t)pos * d, w.embed_positions.scale,
                    m.pos_emb.data(), d);
        for (uint32_t i = 0; i < d; i++) m.hidden[i] += m.pos_emb[i];
        m.rope.ensure(pos);

        for (uint32_t l = 0; l < cfg.n_layers; l++) {
            rmsnorm(m.hidden.data(), w.gamma_attn_f32[l].data(), d, ModelConfig::kRmsEps, m.h_norm.data());
            if (l == 1 && pos == 7) xin_l1 = m.h_norm;
            attention_block(w, cfg, l, m.h_norm.data(), pos, m.kvc[l], m.rope,
                            m.heads.data(), m.attn_out.data(),
                            m.q.data(), m.k.data(), m.v.data(), m.scores.data());
            if (l == 1 && pos == 7) {
                // k_raw: recompute the un-rotated k projection from the saved
                // norm input, then rotate it — must equal the cache contents.
                std::vector<float> kraw(cfg.n_kv_heads * cfg.head_dim);
                gemv_i8_f32(w.layers[l].k, xin_l1.data(), kraw.data());
                for (uint32_t i = 0; i < kraw.size(); i++)
                    CHECK_NEAR(kraw[i], k_raw_g[i], 2e-3f, 1e-3f);
                m.rope.apply(kraw.data(), pos);
                for (uint32_t i = 0; i < kraw.size(); i++) {
                    CHECK_NEAR(kraw[i], k_cached_g[i], 2e-3f, 1e-3f);
                    CHECK_NEAR(kraw[i], m.kvc[l].k_row(pos)[i], 1e-5f, 1e-6f);
                }
                // raw scores of head 0, recomputed from rope-applied q + cache
                const float scale = 1.0f / std::sqrt((float)cfg.head_dim);
                for (uint32_t t = 0; t <= pos; t++) {
                    float s = dot_f32_f32(m.q.data(), m.kvc[l].k_row(t), cfg.head_dim) * scale;
                    CHECK_NEAR(s, scores_g[t], 2e-3f, 1e-3f);
                }
            }
            for (uint32_t i = 0; i < d; i++) m.hidden[i] += m.attn_out[i];
            rmsnorm(m.hidden.data(), w.gamma_ffn_f32[l].data(), d, ModelConfig::kRmsEps, m.ffn_norm.data());
            mlp_block(w, cfg, l, m.ffn_norm.data(), m.mlp_out.data(), m.gate.data(), m.up.data());
            for (uint32_t i = 0; i < d; i++) m.hidden[i] += m.mlp_out[i];
        }
    }
}

TEST(kv_cache_test) {
    KVCache kv;
    kv.init(8, 4);
    const float k[4] = {1, 2, 3, 4}, v[4] = {5, 6, 7, 8};
    kv.write(2, k, v);
    CHECK_EQ(kv.k_row(2)[0], 1.0f);
    CHECK_EQ(kv.k_row(2)[3], 4.0f);
    CHECK_EQ(kv.v_row(2)[2], 7.0f);
    kv.write(3, v, k);  // swap buffers to catch aliasing bugs
    CHECK_EQ(kv.k_row(2)[0], 1.0f);
    CHECK_EQ(kv.k_row(3)[0], 5.0f);
    CHECK_EQ(kv.v_row(3)[1], 2.0f);
    CHECK_EQ(kv.k_row(0)[0], 0.0f);  // untouched slots stay zero
}

TEST(sampler_test) {
    auto logits = read_f32(g_golden + "/sampler_logits.bin");
    auto probs_g = read_f32(g_golden + "/sampler_probs.bin");
    auto greedy_g = read_u32(g_golden + "/sampler_greedy_id.bin");
    auto draws_g = read_u32(g_golden + "/sampler_draws.bin");
    CHECK_EQ(logits.size(), 1024u);
    CHECK_EQ(draws_g.size(), 8u);

    Sampler s(42);
    CHECK_EQ(s.greedy(logits), greedy_g[0]);

    // probability path: softmax(logits / 0.7), float32, same as min_p's internals
    std::vector<float> x = logits;
    const float t = 0.7f;
    for (size_t i = 0; i < x.size(); i++) x[i] /= t;
    softmax(x.data(), x.size());
    for (size_t i = 0; i < x.size(); i++) CHECK_NEAR(x[i], probs_g[i], 1e-4f, 1e-5f);

    // min-p draws must match the Python reference exactly (shared RNG stream)
    Sampler s2(42);
    for (int i = 0; i < 8; i++) CHECK_EQ(s2.min_p(logits, 0.7, 0.1), draws_g[i]);
}

TEST(tokenizer_test) {
    Tokenizer tok;
    CHECK(tok.load(g_tok_path.c_str()));
    CHECK_EQ(tok.entries.size(), 1024u);

    Weights w;
    CHECK(w.load(g_weights_path.c_str()));
    CHECK_EQ(tok.entries.size(), w.cfg.vocab_size);

    // roundtrip: ASCII + multi-byte UTF-8 + raw bytes (byte fallback)
    const std::string s1 = "The meaning of life";
    CHECK(tok.decode(tok.encode(s1)) == s1);
    const std::string s2 = "hello \xe4\xb8\x96\xe7\x95\x8c test";  // "hello 世界 test"
    CHECK(tok.decode(tok.encode(s2)) == s2);
    std::string s3;
    for (int i = 0; i < 256; i++) s3 += (char)i;
    CHECK(tok.decode(tok.encode(s3)) == s3);

    // control tokens: sane ids, produce no text
    CHECK(tok.bos_id < 1024 && tok.eos_id < 1024 && tok.unk_id < 1024 && tok.pad_id < 1024);
    CHECK(tok.decode_id(tok.bos_id).empty());
    CHECK(tok.decode_id(tok.pad_id).empty());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("usage: %s <golden_dir> <weights> <tokenizer>\n", argv[0]);
        return 1;
    }
    g_golden = argv[1];
    g_weights_path = argv[2];
    g_tok_path = argv[3];

    RUN(weights_loader);
    RUN(matmul_i8);
    RUN(matmul_f32);
    RUN(dequant_row_test);
    RUN(rmsnorm_test);
    RUN(rope_test);
    RUN(silu_geglu_test);
    RUN(softmax_test);
    RUN(forward_blocks);
    RUN(attention_scores_kv);
    RUN(kv_cache_test);
    RUN(sampler_test);
    RUN(tokenizer_test);

    if (g_fail == 0)
        std::printf("ALL %d TESTS PASSED\n", g_count);
    else
        std::printf("%d/%d TESTS FAILED (%d assertion failures)\n", g_fail, g_count, g_fail);
    return g_fail ? 1 : 0;
}
