#include "weights.h"

#include <cstring>
#include <string>

namespace gemma {

namespace {

// Offset-walking descriptor builder. Tracks the current byte offset in the
// buffer; each tensor is [rows*cols int8][4-byte scale].
struct Cursor {
    const int8_t* p;
    size_t off = sizeof(fmt::WeightsHeader);

    QTensor next(size_t rows, size_t cols) {
        QTensor t;
        t.data = p + off;
        t.rows = (uint32_t)rows;
        t.cols = (uint32_t)cols;
        off += rows * cols;
        std::memcpy(&t.scale, p + off, sizeof(float));
        off += sizeof(float);
        return t;
    }
};

bool read_whole_file(const char* path, std::vector<char>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "gemma: cannot open weights file: %s\n", path);
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size < 0) {
        std::fclose(f);
        return false;
    }
    out.resize((size_t)size);
    if (size > 0 && std::fread(out.data(), 1, (size_t)size, f) != (size_t)size) {
        std::fprintf(stderr, "gemma: short read on %s\n", path);
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

bool valid_header(const fmt::WeightsHeader& h) {
    if (std::memcmp(h.magic, fmt::kWeightsMagic, 4) != 0) {
        std::fprintf(stderr, "gemma: bad magic in weights file (expected GMW1)\n");
        return false;
    }
    if (h.version != fmt::kWeightsVersion) {
        std::fprintf(stderr, "gemma: unsupported weights version %u (expected %u)\n", h.version, fmt::kWeightsVersion);
        return false;
    }
    for (uint8_t b : h.reserved)
        if (b != 0) {
            std::fprintf(stderr, "gemma: reserved header bytes are non-zero\n");
            return false;
        }
    if (h.n_layers == 0 || h.dim == 0 || h.n_heads == 0 || h.n_kv_heads == 0 || h.mlp_dim == 0 ||
        h.vocab_size == 0 || h.max_seq_len == 0) {
        std::fprintf(stderr, "gemma: weights header has a zero dimension field\n");
        return false;
    }
    if (h.n_kv_heads > h.n_heads || h.n_heads % h.n_kv_heads != 0) {
        std::fprintf(stderr, "gemma: n_heads (%u) must be a multiple of n_kv_heads (%u)\n", h.n_heads, h.n_kv_heads);
        return false;
    }
    if (h.head_dim != h.dim / h.n_heads) {
        std::fprintf(stderr, "gemma: head_dim %u != dim/n_heads = %u\n", h.head_dim, h.dim / h.n_heads);
        return false;
    }
    return true;
}

}  // namespace

size_t Weights::expected_file_size(const ModelConfig& cfg) {
    const size_t kv = (size_t)cfg.n_kv_heads * cfg.head_dim;
    size_t n = sizeof(fmt::WeightsHeader);
    n += (size_t)cfg.vocab_size * cfg.dim + 4;          // embed_tokens
    n += (size_t)cfg.max_seq_len * cfg.dim + 4;         // embed_positions
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        n += 2 * ((size_t)cfg.dim * cfg.dim + 4);       // q, o
        n += 2 * (kv * cfg.dim + 4);                    // k, v (GQA: only kv heads)
        n += 2 * ((size_t)cfg.mlp_dim * cfg.dim + 4);   // gate, up
        n += (size_t)cfg.dim * cfg.mlp_dim + 4;         // down
        n += 2 * ((size_t)cfg.dim + 4);                 // gammas
    }
    n += (size_t)cfg.dim + 4;                           // final_norm_gamma
    n += (size_t)cfg.vocab_size * cfg.dim + 4;          // lm_head
    return n;
}

bool Weights::load(const char* path) {
    std::vector<char> file;
    if (!read_whole_file(path, file)) return false;

    if (file.size() < sizeof(fmt::WeightsHeader)) {
        std::fprintf(stderr, "gemma: weights file too small for header: %s\n", path);
        return false;
    }

    fmt::WeightsHeader h;
    std::memcpy(&h, file.data(), sizeof(h));
    if (!valid_header(h)) return false;

    ModelConfig c;
    c.n_layers = h.n_layers;
    c.dim = h.dim;
    c.n_heads = h.n_heads;
    c.n_kv_heads = h.n_kv_heads;
    c.mlp_dim = h.mlp_dim;
    c.vocab_size = h.vocab_size;
    c.max_seq_len = h.max_seq_len;
    c.head_dim = h.head_dim;
    c.rope_base = h.rope_base;

    size_t expected = expected_file_size(c);
    if (file.size() != expected) {
        std::fprintf(stderr, "gemma: weights file size %zu != expected %zu (shape mismatch or truncated file)\n",
                     file.size(), expected);
        return false;
    }

    if (!buf.alloc(file.size())) {
        std::fprintf(stderr, "gemma: out of memory allocating %zu bytes for weights\n", file.size());
        return false;
    }
    std::memcpy(buf.ptr, file.data(), file.size());
    cfg = c;

    // Walk the tensor section and build descriptors.
    Cursor cur{(const int8_t*)buf.ptr};
    embed_tokens = cur.next(cfg.vocab_size, cfg.dim);
    embed_positions = cur.next(cfg.max_seq_len, cfg.dim);
    const uint32_t kv = cfg.n_kv_heads * cfg.head_dim;
    layers.resize(cfg.n_layers);
    gamma_attn_f32.resize(cfg.n_layers);
    gamma_ffn_f32.resize(cfg.n_layers);
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        Layer& L = layers[l];
        L.q = cur.next(cfg.dim, cfg.dim);
        L.k = cur.next(kv, cfg.dim);
        L.v = cur.next(kv, cfg.dim);
        L.o = cur.next(cfg.dim, cfg.dim);
        L.gate = cur.next(cfg.mlp_dim, cfg.dim);
        L.up = cur.next(cfg.mlp_dim, cfg.dim);
        L.down = cur.next(cfg.dim, cfg.mlp_dim);
        L.gamma_attn = cur.next(cfg.dim, 1);
        L.gamma_ffn = cur.next(cfg.dim, 1);
        gamma_attn_f32[l].resize(cfg.dim);
        gamma_ffn_f32[l].resize(cfg.dim);
        for (uint32_t i = 0; i < cfg.dim; i++) {
            gamma_attn_f32[l][i] = (float)L.gamma_attn.data[i] * L.gamma_attn.scale;
            gamma_ffn_f32[l][i] = (float)L.gamma_ffn.data[i] * L.gamma_ffn.scale;
        }
    }
    final_norm_gamma = cur.next(cfg.dim, 1);
    final_norm_gamma_f32.resize(cfg.dim);
    for (uint32_t i = 0; i < cfg.dim; i++)
        final_norm_gamma_f32[i] = (float)final_norm_gamma.data[i] * final_norm_gamma.scale;
    lm_head = cur.next(cfg.vocab_size, cfg.dim);
    return true;
}

void Weights::dump(FILE* f) const {
    std::fprintf(f, "model config:\n");
    std::fprintf(f, "  n_layers    = %u\n", cfg.n_layers);
    std::fprintf(f, "  dim         = %u\n", cfg.dim);
    std::fprintf(f, "  n_heads     = %u (kv: %u)\n", cfg.n_heads, cfg.n_kv_heads);
    std::fprintf(f, "  head_dim    = %u\n", cfg.head_dim);
    std::fprintf(f, "  mlp_dim     = %u\n", cfg.mlp_dim);
    std::fprintf(f, "  vocab_size  = %u\n", cfg.vocab_size);
    std::fprintf(f, "  max_seq_len = %u\n", cfg.max_seq_len);
    std::fprintf(f, "  rope_base   = %.4g\n", cfg.rope_base);
    std::fprintf(f, "  file size   = %zu bytes\n", buf.size);
    std::fprintf(f, "tensors:\n");
    auto one = [&](const char* name, const QTensor& t) {
        std::fprintf(f, "  %-24s [%ux%u] scale=%.6g\n", name, t.rows, t.cols, t.scale);
    };
    one("embed_tokens", embed_tokens);
    one("embed_positions", embed_positions);
    for (uint32_t l = 0; l < cfg.n_layers; l++) {
        char name[64];
        std::snprintf(name, sizeof(name), "layer %u: q_proj", l);
        one(name, layers[l].q);
        std::snprintf(name, sizeof(name), "layer %u: k_proj", l);
        one(name, layers[l].k);
        std::snprintf(name, sizeof(name), "layer %u: v_proj", l);
        one(name, layers[l].v);
        std::snprintf(name, sizeof(name), "layer %u: o_proj", l);
        one(name, layers[l].o);
        std::snprintf(name, sizeof(name), "layer %u: gate_proj", l);
        one(name, layers[l].gate);
        std::snprintf(name, sizeof(name), "layer %u: up_proj", l);
        one(name, layers[l].up);
        std::snprintf(name, sizeof(name), "layer %u: down_proj", l);
        one(name, layers[l].down);
        std::snprintf(name, sizeof(name), "layer %u: gamma_attn", l);
        one(name, layers[l].gamma_attn);
        std::snprintf(name, sizeof(name), "layer %u: gamma_ffn", l);
        one(name, layers[l].gamma_ffn);
    }
    one("final_norm_gamma", final_norm_gamma);
    one("lm_head", lm_head);
}

}  // namespace gemma
