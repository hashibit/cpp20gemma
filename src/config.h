// Model configuration and binary file header definitions.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace gemma {

// ---------------------------------------------------------------------------
// Model configuration (data-driven; the weights file header is authoritative)
// ---------------------------------------------------------------------------
struct ModelConfig {
    uint32_t n_layers    = 0;
    uint32_t dim         = 0;   // embedding / hidden dimension
    uint32_t n_heads     = 0;   // query heads
    uint32_t n_kv_heads  = 0;   // KV heads (GQA)
    uint32_t mlp_dim     = 0;   // intermediate FFN dimension
    uint32_t vocab_size  = 0;
    uint32_t max_seq_len = 0;   // rows in embed_positions
    uint32_t head_dim    = 0;   // dim / n_heads
    float    rope_base   = 10000.0f;

    // Gemma-style RMSNorm epsilon (fixed constant, matches Gemma 1/2/3).
    static constexpr float kRmsEps = 1e-6f;
};

// ---------------------------------------------------------------------------
// Named presets. "270m" / "1B" are the larger-scale fixtures.
// (The table's param counts don't line up with real 270M/1B Gemma checkpoints;
//  these are the engine's presets — real-weight conversion validates shapes.)
// ---------------------------------------------------------------------------
struct ModelPreset {
    const char* name;
    ModelConfig cfg;
};

inline const ModelPreset& preset_for(const std::string& name) {
    static const ModelPreset presets[] = {
        {"270m", {26, 1536, 4, 1, 6144, 0, 128 * 1024, 384, 10000.0f}},
        {"1B",   {28, 2048, 4, 1, 8192, 0, 128 * 1024, 512, 10000.0f}},
        {"tiny", {2,  256,  4, 1, 1024, 1024,      256,      64,  10000.0f}},
    };
    for (const auto& p : presets)
        if (name == p.name) return p;
    // Unknown name -> the preset lookups return an invalid marker instead of
    // erroring out, so the caller can decide how strict to be.
    static const ModelPreset invalid = {"<invalid>", {}};
    return invalid;
}

// ---------------------------------------------------------------------------
// Weights binary format v1 (little-endian)
// ---------------------------------------------------------------------------
namespace fmt {
constexpr char kWeightsMagic[4] = {'G', 'M', 'W', '1'};
constexpr uint32_t kWeightsVersion = 1;
constexpr char kTokenizerMagic[4] = {'G', 'M', 'T', '1'};
constexpr uint32_t kTokenizerVersion = 1;

// 64-byte fixed header; tensor data follows immediately.
#pragma pack(push, 1)
struct WeightsHeader {
    char     magic[4];         // "GMW1"
    uint32_t version;          // 1
    uint32_t n_layers;
    uint32_t dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t mlp_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;      // rows in embed_positions
    uint32_t head_dim;         // = dim / n_heads (stored for validation)
    float    rope_base;
    uint8_t  reserved[20];     // must be zero
};

// 48-byte fixed header for the tokenizer file.
struct TokenizerHeader {
    char     magic[4];         // "GMT1"
    uint32_t version;          // 1
    uint32_t vocab_size;
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t unk_id;
    uint32_t pad_id;
    uint8_t  reserved[20];     // must be zero
};
#pragma pack(pop)

static_assert(sizeof(WeightsHeader) == 64, "weights header must be 64 bytes");
static_assert(sizeof(TokenizerHeader) == 48, "tokenizer header must be 48 bytes");

// Token types in the tokenizer file.
enum TokenType : uint32_t { kNormal = 0, kByte = 1, kControl = 2 };
}  // namespace fmt

// ---------------------------------------------------------------------------
// Little helpers shared across the codebase
// ---------------------------------------------------------------------------
inline void fail(const char* msg) {
    std::fprintf(stderr, "gemma: %s\n", msg);
    std::exit(1);
}

inline void warn(const char* msg) {
    std::fprintf(stderr, "gemma: warning: %s\n", msg);
}

}  // namespace gemma
