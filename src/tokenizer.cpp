#include "tokenizer.h"

#include <cstdio>
#include <cstring>

#include "config.h"

namespace gemma {

namespace {

bool read_whole_file(const char* path, std::vector<char>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "gemma: cannot open tokenizer file: %s\n", path);
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

}  // namespace

bool Tokenizer::load(const char* path) {
    std::vector<char> file;
    if (!read_whole_file(path, file)) return false;
    if (file.size() < sizeof(fmt::TokenizerHeader)) {
        std::fprintf(stderr, "gemma: tokenizer file too small: %s\n", path);
        return false;
    }

    fmt::TokenizerHeader h;
    std::memcpy(&h, file.data(), sizeof(h));
    if (std::memcmp(h.magic, fmt::kTokenizerMagic, 4) != 0) {
        std::fprintf(stderr, "gemma: bad magic in tokenizer file (expected GMT1)\n");
        return false;
    }
    if (h.version != fmt::kTokenizerVersion) {
        std::fprintf(stderr, "gemma: unsupported tokenizer version %u\n", h.version);
        return false;
    }
    for (uint8_t b : h.reserved)
        if (b != 0) {
            std::fprintf(stderr, "gemma: tokenizer reserved header bytes are non-zero\n");
            return false;
        }

    bos_id = h.bos_id;
    eos_id = h.eos_id;
    unk_id = h.unk_id;
    pad_id = h.pad_id;
    for (uint32_t i = 0; i < 256; i++) byte_token[i] = UINT32_MAX;

    entries.resize(h.vocab_size);
    size_t off = sizeof(h);
    size_t blob_len = 0;
    for (uint32_t i = 0; i < h.vocab_size; i++) {
        if (off + 12 > file.size()) {
            std::fprintf(stderr, "gemma: tokenizer file truncated at entry %u\n", i);
            return false;
        }
        uint32_t len = 0, type = 0;
        float score = 0.0f;
        std::memcpy(&len, file.data() + off, 4);
        std::memcpy(&type, file.data() + off + 4, 4);
        std::memcpy(&score, file.data() + off + 8, 4);
        off += 12;
        if (off + len > file.size()) {
            std::fprintf(stderr, "gemma: tokenizer piece %u overruns the file\n", i);
            return false;
        }
        entries[i].type = type;
        entries[i].score = score;
        blob_len += len;
        if (len > max_piece_len) max_piece_len = len;
        off += len;
    }

    blob.reserve(blob_len);
    off = sizeof(h);
    for (uint32_t i = 0; i < h.vocab_size; i++) {
        uint32_t len = 0;
        std::memcpy(&len, file.data() + off, 4);
        off += 12;
        std::string_view piece(file.data() + off, len);
        off += len;
        size_t at = blob.size();
        blob.append(piece.data(), piece.size());
        entries[i].piece = std::string_view(blob.data() + at, len);
        auto [it, inserted] = piece_to_id.emplace(entries[i].piece, i);
        if (!inserted)
            warn("tokenizer has duplicate pieces; longest-match may be ambiguous");
        if (entries[i].type == fmt::kByte && len == 1)
            byte_token[(uint8_t)entries[i].piece[0]] = i;
    }

    uint32_t missing = 0;
    for (uint32_t i = 0; i < 256; i++)
        if (byte_token[i] == UINT32_MAX) missing++;
    if (missing > 0)
        warn("tokenizer is missing byte-fallback tokens; some bytes will map to <unk>");
    return true;
}

std::vector<uint32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<uint32_t> ids;
    size_t i = 0;
    while (i < text.size()) {
        size_t L = max_piece_len < text.size() - i ? max_piece_len : text.size() - i;
        bool matched = false;
        for (; L > 0; L--) {
            auto it = piece_to_id.find(std::string_view(text.data() + i, L));
            if (it != piece_to_id.end()) {
                ids.push_back(it->second);
                i += L;
                matched = true;
                break;
            }
        }
        if (!matched) {
            uint32_t bt = byte_token[(uint8_t)text[i]];
            ids.push_back(bt == UINT32_MAX ? unk_id : bt);
            i += 1;
        }
    }
    return ids;
}

std::string Tokenizer::decode_id(uint32_t id) const {
    if (id >= entries.size()) return "";
    const Entry& e = entries[id];
    if (e.type == fmt::kControl) return "";  // control tokens produce no text
    return std::string(e.piece);
}

std::string Tokenizer::decode(const std::vector<uint32_t>& ids) const {
    std::string out;
    for (uint32_t id : ids) out += decode_id(id);
    return out;
}

}  // namespace gemma
