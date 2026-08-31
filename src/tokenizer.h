// Binary tokenizer (GMT1 format): greedy longest-prefix encode + byte fallback.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gemma {

struct Tokenizer {
    struct Entry {
        uint32_t type = 0;  // fmt::TokenType
        float score = 0.0f;
        std::string_view piece;  // view into `blob` below
    };

    uint32_t bos_id = 0, eos_id = 0, unk_id = 0, pad_id = 0;
    std::string blob;              // all pieces concatenated (may embed NULs)
    std::vector<Entry> entries;    // [vocab_size]
    std::unordered_map<std::string_view, uint32_t> piece_to_id;
    uint32_t byte_token[256];      // byte value -> token id (byte-fallback)
    uint32_t max_piece_len = 0;

    bool load(const char* path);

    // Greedy longest-prefix encode; unmatched bytes fall back to byte tokens.
    std::vector<uint32_t> encode(const std::string& text) const;
    // Control tokens (type 2) are dropped; byte tokens append their raw byte.
    std::string decode(const std::vector<uint32_t>& ids) const;
    std::string decode_id(uint32_t id) const;
};

}  // namespace gemma
